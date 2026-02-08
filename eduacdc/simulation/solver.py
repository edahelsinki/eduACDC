"""
Simulation solver for ACDC systems.
"""

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.integrate

from ..core.coagulation_loss import CoagulationLossCalculator
from ..core.equations import ClusterEquations
from ..core.system import SimulationSystem
from ..core.wall_loss import WallLossCalculator
from ..utils import parse_quantity, ureg
from .results import SimulationResults

# Set up logging

logger = logging.getLogger(__name__)

def parse_initial_conditions(
        system: SimulationSystem, initial_conditions: Dict[str, Dict[str, str | float]]
    ) -> Tuple[np.ndarray, np.ndarray, set, dict|None]:
        """
        Parse initial conditions from a dictionary of initial conditions.
        Example:
        initial_conditions = {
            "1SA": {"type": "constant", "value": "1e6 cm^-3"}
            "1SA": {"type": "source", "value": "1e6 cm^-3/s"}
            "1SA": {"type": "initial", "value": "1e6 cm^-3"}
        } # type: Dict[str, Dict[str, str | float]]
        Args:
            initial_conditions: Dictionary of initial conditions (see example above).
            The key is the cluster label, and the value is a dictionary with the source type and value.
            The source type is a string, and the value is a float or string.
            The value is a string in [volume]^-3 or ppt, ppb, ppm, etc.
        
        Raises:
            ValueError: If cluster not found in system.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, set, dict]: Tuple of initial concentrations, sources, const_clusters, and sum_const_clusters.
            initial_concentrations: Array of initial concentrations in m^-3.
            sources: Array of sources in m^-3/s.
            const_clusters: Set of cluster indices with constant concentrations.
            sum_const_clusters: Dictionary of cluster indices with sum of constant concentrations.
                The key is the cluster index, and the value is a dictionary with the value and independent clusters.
        """
        initial_concentrations = np.zeros(system.n_clusters)
        sources = np.zeros(system.n_clusters)
        const_clusters = set() # clusters with constant concentrations
        sum_const_clusters = dict() # clusters with sum of constant concentrations
        temperature = system.conditions.temperature * ureg.kelvin
        # Assumption is that conversions from ppt are done to 1 atm pressure
        pressure = ureg("1 atm")
        with ureg.context("conc", temperature=temperature, pressure=pressure):
            for key in initial_conditions.keys():
                initial_condition_type = initial_conditions[key].get("type")
                initial_condition_value = initial_conditions[key].get("value")
                
                idx = system.clusters.get_index_by_label(key)
                # check if cluster is found
                if idx is None:
                    raise ValueError(f"Cluster {key} not found in system. Available clusters: {system.clusters.get_labels()}")
                
                if initial_condition_type == "constant":
                    # convert value to SI units
                    _value = parse_quantity(
                        initial_condition_value, default_unit="cm^-3", target_unit="m^-3"
                    )
                    initial_concentrations[idx] = _value
                    const_clusters.add(idx)
                    logger.info(
                        f"Initial condition for cluster {key}: idx= {idx}, type= {initial_condition_type}, value= {initial_condition_value}, parsed value= {_value:g} m^-3"
                    )
                elif initial_condition_type == "source":
                    # convert value to SI units
                    _value = parse_quantity(
                        initial_condition_value, default_unit="cm^-3/s", target_unit="m^-3/s"
                    )
                    sources[idx] = _value
                    logger.info(
                        f"Initial condition for cluster {key}: idx= {idx}, type= {initial_condition_type}, value= {initial_condition_value}, parsed value= {_value:g} m^-3/s"
                    )
                elif initial_condition_type == "initial":
                    # convert value to SI units
                    _value = parse_quantity(
                        initial_condition_value, default_unit="cm^-3", target_unit="m^-3"
                    )
                    initial_concentrations[idx] = _value
                    logger.info(
                        f"Initial condition for cluster {key}: idx= {idx}, type= {initial_condition_type}, value= {initial_condition_value}, parsed value= {_value:g} m^-3"
                    )
                elif initial_condition_type == "sum_constant":
                    # convert value to SI units
                    _value = parse_quantity(
                        initial_condition_value, default_unit="cm^-3", target_unit="m^-3"
                    )
                    independent_clusters_raw = initial_conditions[key].get("independent_clusters")
                    if independent_clusters_raw is None or not isinstance(independent_clusters_raw, list):
                        raise ValueError(f"Independent clusters not specified for cluster {key}")
                    independent_clusters: list[str] = [
                        str(cluster_label) for cluster_label in independent_clusters_raw
                    ]

                    sum_const_clusters[idx] = {
                        "value": _value,
                        "independent_clusters": [system.clusters.get_index_by_label(cluster_label) for cluster_label in independent_clusters]
                    }
                    logger.info(
                        f"Initial condition for cluster {key}: idx= {idx}, type= {initial_condition_type}, value= {initial_condition_value}, parsed value= {_value:g} m^-3, independent clusters= {independent_clusters}"
                    )
        # check if sum_const_clusters is empty
        if not sum_const_clusters:
            sum_const_clusters = None
        else:
            for idx in sum_const_clusters.keys():
                _value = sum_const_clusters[idx]["value"]
                independent_clusters = sum_const_clusters[idx]["independent_clusters"]
                # set initial concentration to ensure constraint is satisfied
                initial_concentrations[idx] = _value - sum(initial_concentrations[independent_clusters])
                logger.debug(
                    f"Constant sum cluster: {system.clusters.get_label(idx)} idx= {idx}, initial concentration= {initial_concentrations[idx]:g} m^-3"
                )
        return initial_concentrations, sources, const_clusters, sum_const_clusters

class Simulation:
    """Simulation solver using SymPy equations."""

    def __init__(
        self,
        system: SimulationSystem,
        equations: ClusterEquations,
    ):
        logger.info("Starting Simulation initialization")
        start_time = time.time()
        self.system: SimulationSystem = system
        self.equations: ClusterEquations = equations
        assert self.system.n_clusters == self.equations.n_clusters, "Number of clusters in system and equations must match"
        total_time = time.time() - start_time
        logger.debug(f"Simulation initialization completed in {total_time:.4f}s")

    def update_equations_config(self, **kwargs):
        """
        Update the equations configuration with new parameters (e.g., disable_nonmonomers),
        and recreate the equations accordingly.
        """
        self.equations.update_equations_config(**kwargs)

    def run(
        self,
        initial_conditions: Dict[str, Dict[str, str | float]],
        t_span: Tuple[float, float] = (0.0, 100.0),
        relative_humidity: Optional[float] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> SimulationResults:
        """
        Run the simulation using the simulation configuration.

        Args:
            initial_conditions (Dict[str, Dict[str, str | float]]): Initial conditions.
            t_span (Tuple[float, float], optional): Time span (t_start, t_end) in seconds. Defaults to (0.0, 100.0).
            relative_humidity (Optional[float], optional): Relative humidity (0-1). If provided, updates system conditions.
            temperature (Optional[float], optional): Temperature in K. If provided, updates system conditions.
            **kwargs: Additional arguments for the simulation.

        Returns:
            SimulationResults: Simulation results.
        """
        if relative_humidity is not None:
            self.system.update_conditions(relative_humidity=relative_humidity)
        if temperature is not None:
            self.system.update_conditions(temperature=temperature)

        initial_concentrations, sources, const_clusters, sum_const_clusters = parse_initial_conditions(self.system, initial_conditions)
        results = self._run(
            t_span=t_span,
            initial_concentrations=initial_concentrations,
            sources=sources,
            const_clusters=const_clusters,
            sum_const_clusters=sum_const_clusters,
            **kwargs,
        )
        results["initial_conditions"] = initial_conditions
        return SimulationResults(
            system=self.system,         
            equations=self.equations,
            **results,
        )

    def _run(
        self,
        t_span: Tuple[float, float] = (0.0, 100.0),
        initial_concentrations: Optional[np.ndarray] = None,
        sources: Optional[np.ndarray] = None,
        const_clusters: Optional[set] = None,
        sum_const_clusters: Optional[dict] = None,
        wall_loss_calculator: WallLossCalculator | None = None,
        coagulation_loss_calculator: CoagulationLossCalculator | None = None,
        dilution_loss: float = 0.0,
        method: str = "BDF",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        compute_formation_rates: bool = True,
        compute_final_outflux: bool = False,
        compute_net_fluxes: bool = False,
        calc_final_sources: bool = True,
        **kwargs,
    ) -> dict:
        """
        Run the simulation using SymPy-generated equations.

        Args:
            t_span: Time span (t_start, t_end) in seconds
            initial_concentrations: Initial cluster concentrations (m⁻³). Shape: (n_clusters,)
            sources: Source terms for each cluster (m⁻³ s⁻¹). Shape: (n_clusters,)
            const_clusters: Set of cluster indices with constant concentrations.
            sum_const_clusters: Dictionary of cluster indices with sum of constant concentrations.
            wall_loss_calculator: Wall loss calculator. Defaults to constant 0.0.
            coagulation_loss_calculator: Coagulation loss calculator. Defaults to constant 0.0.
            method: ODE solver method
            rtol: Relative tolerance
            atol: Absolute tolerance
            compute_formation_rates: Whether to compute time-dependent formation rates during simulation
            compute_final_outflux: Whether to compute the final outflux rates
            compute_net_fluxes: Whether to compute the net fluxes
            calc_final_sources: Whether to compute the final sources when computing net fluxes
        Returns:
            dict: Dictionary of results
                time: Time points (s)
                concentrations: Concentrations (m⁻³)
                collision_coefficients: Collision rates (m⁻³ s⁻¹)
                evaporation_coefficients: Evaporation rates (m⁻³ s⁻¹)
                wall_loss_rates: Wall loss rates (m⁻³ s⁻¹)
                coagulation_sink_rates: Coagulation sink rates (m⁻³ s⁻¹)
                formation_rates: Formation rates (m⁻³ s⁻¹)
                final_outflux_rates: Final outflux rates (m⁻³ s⁻¹)
                net_fluxes: Net fluxes (m⁻³ s⁻¹)
                final_sources: Final sources (m⁻³ s⁻¹)
        """
        logger.info(f"Starting simulation run with t_span={t_span}, method={method}")
        run_start = time.time()


        n_clusters = self.system.n_clusters
        logger.debug(f"Number of clusters: {n_clusters}")

        # Set default initial concentrations if not provided
        if initial_concentrations is None:
            initial_concentrations = np.zeros(n_clusters)
            logger.debug("Using zero initial concentrations")

        # Set default sources if not provided
        if sources is None:
            sources = np.zeros(n_clusters)
            logger.debug("Using zero sources")

        # Set default const_clusters if not provided
        if const_clusters is None:
            const_clusters = set()
            logger.debug("Using variable concentrations for all clusters")
        # Set default sum_const_clusters if not provided
        if sum_const_clusters is None:
            sum_const_clusters = dict()
            logger.debug("Using no sum of constant clusters")
        # Validate inputs
        if len(initial_concentrations) != n_clusters:
            raise ValueError(f"Initial concentrations must have length {n_clusters}")
        if len(sources) != n_clusters:
            raise ValueError(f"Sources must have length {n_clusters}")
        if len(const_clusters) > n_clusters:
            raise ValueError(f"const_clusters must have length less than or equal to {n_clusters}")
        if sum_const_clusters is not None:
            if len(sum_const_clusters) > n_clusters:
                raise ValueError(f"sum_const_clusters must have length less than or equal to {n_clusters}")
    
        logger.debug("Getting reaction rates...")
        rates_start = time.time()

        # Get reaction rates
        collision_coefficients = self.system.get_collision_coefficients()
        evaporation_coefficients = self.system.get_evaporation_coefficients()
        rates_time = time.time() - rates_start

        logger.debug(f"Reaction rates calculated in {rates_time:.3f}s")

        external_loss_rates_start = time.time()
        # set default wall loss and coagulation loss calculators if not provided
        if wall_loss_calculator is None:
            wall_loss_calculator = WallLossCalculator.create_constant(0.0)
        if coagulation_loss_calculator is None:
            coagulation_loss_calculator = CoagulationLossCalculator.create_constant(0.0)
        # Calculate wall loss and coagulation loss rates
        wall_loss_rates = wall_loss_calculator.calculate_coefficients(
            clusters=self.system.clusters,
            cluster_properties=self.system.cluster_properties,
            conditions=self.system.conditions
        )
        coagulation_sink_rates = coagulation_loss_calculator.calculate_coefficients(
            clusters=self.system.clusters,
            cluster_properties=self.system.cluster_properties,
            conditions=self.system.conditions
        )
        
        external_loss_rates_time = time.time() - external_loss_rates_start
        logger.debug(f"External loss rates calculated in {external_loss_rates_time:.3f}s")

        # Create ODE function using SymPy equations
        ode_function = self.equations.get_ode_function(
            collision_coefficients=collision_coefficients,
            evaporation_coefficients=evaporation_coefficients,
            wall_loss_rates=wall_loss_rates,
            coagulation_sink_rates=coagulation_sink_rates,
            sources=sources,
            const_clusters=const_clusters,
            sum_const_clusters=sum_const_clusters,
            dilution_loss=dilution_loss,
        )

        logger.info("Solving ODE system...")
        solve_start = time.time()

        # Solve the ODE system
        solution = scipy.integrate.solve_ivp(
            fun=ode_function,
            t_span=t_span,
            y0=initial_concentrations,
            method=method,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

        solve_time = time.time() - solve_start
        logger.info(f"ODE system solved in {solve_time:.3f}s")

        logger.debug("Processing results...")
        results_start = time.time()

        # Transpose solution.y to shape (n_time_points, n_clusters)
        concentrations = solution.y.T

        # Calculate outgrowth rates if requested
        formation_rates = None
        if compute_formation_rates:
            logger.debug("Calculating formation rates...")
            formation_start = time.time()
            formation_rates = self.equations.compute_formation_rate(concentrations, collision_coefficients)
            formation_time = time.time() - formation_start
            logger.debug(f"Formation rates calculated in {formation_time:.3f}s")

        final_outflux_rates = None
        if compute_final_outflux:
            logger.debug("Calculating outflux...")
            outflux_start = time.time()
            final_outflux_rates = self.equations.compute_outflux_matrix(concentrations[-1], collision_coefficients)
            outflux_time = time.time() - outflux_start
            logger.debug(f"Outflux calculated in {outflux_time:.3f}s")

        all_net_fluxes = None
        final_sources = None
        if compute_net_fluxes:
            fluxes_start = time.time()
            fluxes_fn = self.equations.get_net_fluxes_function()
            # calculate net fluxes on final concentrations (steady state)
            all_net_fluxes, final_sources = fluxes_fn(
                concentrations[-1],
                collision_coefficients,
                evaporation_coefficients,
                wall_loss_rates,
                coagulation_sink_rates,
                sources,
                dilution_loss,
                calc_final_sources=calc_final_sources,
            )
            
            fluxes_time = time.time() - fluxes_start
            logger.debug(f"Fluxes calculated in {fluxes_time:.3f}s")
        # Create results object
        results = {
            "time": solution.t,
            "concentrations": concentrations,
            "collision_coefficients": collision_coefficients,
            "evaporation_coefficients": evaporation_coefficients,
            "wall_loss_rates": wall_loss_rates,
            "coagulation_sink_rates": coagulation_sink_rates,
            "formation_rates": formation_rates,
            "final_outflux_rates": final_outflux_rates,
            "net_fluxes": all_net_fluxes,
            "final_sources": final_sources,
        }

        results_time = time.time() - results_start
        logger.debug(f"Results processed in {results_time:.3f}s")

        total_run_time = time.time() - run_start
        logger.info(f"Simulation run completed in {total_run_time:.3f}s")

        return results


    def print_equations(self, max_terms: int = 5):
        """Print the symbolic equations."""
        self.equations.print_ode_equations(max_terms)

    def run_steady_state(
        self,
        initial_conditions: Dict[str, Dict[str, str | float]],
        max_time: float = 1e8,
        tolerance: float = 1e-6,
        min_concentration_threshold: float = 1e-6,
        dilution_loss: float = 0.0,
        compute_formation_rates: bool = True,
        compute_final_outflux: bool = False,
        compute_net_fluxes: bool = False,
        calc_final_sources: bool = True,
        wall_loss_calculator: WallLossCalculator | None = None,
        coagulation_loss_calculator: CoagulationLossCalculator | None = None,
        relative_humidity: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> SimulationResults:
        """
        Find steady-state concentrations using SymPy equations.

        Args:
            initial_conditions: Initial conditions.
            max_time: Maximum time to run the simulation
            tolerance: Convergence tolerance
            min_concentration_threshold: Minimum concentration threshold for steady state calculation
            compute_formation_rates: Whether to compute formation rates during simulation
            compute_final_outflux: Whether to compute the final outflux rates
            compute_net_fluxes: Whether to compute the net fluxes
            calc_final_sources: Whether to compute the final sources when computing net fluxes
            wall_loss_calculator: Wall loss calculator. Defaults to constant 0.0.
            coagulation_loss_calculator: Coagulation loss calculator. Defaults to constant 0.0.
            relative_humidity (Optional[float], optional): Relative humidity (0-1). If provided, updates system conditions.
            temperature (Optional[float], optional): Temperature in K. If provided, updates system conditions.
        Returns:
            SimulationResults object with steady-state concentrations
        """
        logger.info(
            f"Starting steady state calculation with max_time={max_time}, tolerance={tolerance}"
        )
        steady_start = time.time()

        if relative_humidity is not None:
            self.system.update_conditions(relative_humidity=relative_humidity)
        if temperature is not None:
            self.system.update_conditions(temperature=temperature)

        n_clusters = self.system.n_clusters
        logger.debug(f"Number of clusters: {n_clusters}")

        if wall_loss_calculator is None:
            wall_loss_calculator = WallLossCalculator.create_constant(0.0)
        if coagulation_loss_calculator is None:
            coagulation_loss_calculator = CoagulationLossCalculator.create_constant(0.0)
        initial_concentrations, sources, const_clusters, sum_const_clusters = parse_initial_conditions(self.system, initial_conditions)
        logger.debug("Getting reaction rates...")
        rates_start = time.time()

        # Get reaction rates
        collision_coefficients = self.system.get_collision_coefficients()
        evaporation_coefficients = self.system.get_evaporation_coefficients()
        
        rates_time = time.time() - rates_start
        logger.debug(f"Reaction rates calculated in {rates_time:.3f}s")

        # Calculate external loss rates
        external_loss_rates_start = time.time()
        wall_loss_rates = wall_loss_calculator.calculate_coefficients(
            clusters=self.system.clusters,
            cluster_properties=self.system.cluster_properties,
            conditions=self.system.conditions
        )
        coagulation_sink_rates = coagulation_loss_calculator.calculate_coefficients(
            clusters=self.system.clusters,
            cluster_properties=self.system.cluster_properties,
            conditions=self.system.conditions
        )
        external_loss_rates_time = time.time() - external_loss_rates_start
        logger.debug(f"External loss rates calculated in {external_loss_rates_time:.3f}s")


        logger.debug("Creating ODE function...")
        ode_start = time.time()

        # Create ODE function
        ode_function = self.equations.get_ode_function(
            collision_coefficients=collision_coefficients,
            evaporation_coefficients=evaporation_coefficients,
            wall_loss_rates=wall_loss_rates,
            coagulation_sink_rates=coagulation_sink_rates,
            sources=sources,
            const_clusters=const_clusters,
            sum_const_clusters=sum_const_clusters,
            dilution_loss=dilution_loss,
        )
        
        ode_time = time.time() - ode_start
        logger.debug(f"ODE function created in {ode_time:.3f}s")

        # Initial guess: all concentrations zero except sources
        concentrations = initial_concentrations.copy()

        logger.info("Starting solver for steady state...")
        iteration_start = time.time()

        bdf_solver = scipy.integrate.BDF(
            fun=ode_function,
            t0=0.0,
            t_bound=max_time,
            y0=initial_concentrations,
        )
        t = [0.0]
        concentrations = [initial_concentrations.copy()]
        old_concentrations = concentrations[-1].copy()
        while bdf_solver.status == "running":
            C = bdf_solver.y
            converged = False
            if len(t) > 1 and np.max(np.abs(C-old_concentrations)/np.maximum(C,min_concentration_threshold)) < tolerance:
                converged = True
            # check if constraints are satisfied
            if sum_const_clusters:
                for cluster_idx, constraint_data in sum_const_clusters.items():
                    independent_clusters = constraint_data["independent_clusters"]
                    fixed_concentration = constraint_data["value"]
                    total_sum = sum(C[independent_clusters]) + C[cluster_idx]
                    if abs(fixed_concentration - total_sum)/fixed_concentration > tolerance:
                        scale_factor = fixed_concentration/total_sum
                        C[cluster_idx] = fixed_concentration - total_sum
                        C[independent_clusters] = C[independent_clusters]*scale_factor
                        converged = False
            if converged:
                break
            # update values
            t.append(bdf_solver.t)
            concentrations.append(C.copy())
            old_concentrations = concentrations[-1].copy()
            # next step
            bdf_solver.step()

        if bdf_solver.status == "failed":
            logger.warning("Steady state did not converge after max_time")
            raise RuntimeError("Steady state did not converge after max_time")
        
        t = np.array(t)
        concentrations = np.array(concentrations)

        iteration_time = time.time() - iteration_start
        logger.info(f"Steady state solution after {t[-1]:.3e}s completed in {iteration_time:.3f}s")

        # Calculate outgrowth rates if requested
        formation_rates = None
        if compute_formation_rates:
            logger.debug("Calculating formation rates...")
            formation_start = time.time()
            formation_rates = self.equations.compute_formation_rate(concentrations, collision_coefficients)
            formation_time = time.time() - formation_start
            logger.debug(f"Formation rates calculated in {formation_time:.3f}s")

        final_outflux_rates = None
        if compute_final_outflux:
            logger.debug("Calculating outflux...")
            outflux_start = time.time()
            final_outflux_rates = self.equations.compute_outflux_matrix(concentrations[-1], collision_coefficients)
            outflux_time = time.time() - outflux_start
            logger.debug(f"Outflux calculated in {outflux_time:.3f}s")

        all_net_fluxes = None
        final_sources = None
        if compute_net_fluxes:
            fluxes_start = time.time()
            fluxes_fn = self.equations.get_net_fluxes_function()
            # calculate net fluxes on final concentrations (steady state)
            all_net_fluxes, final_sources = fluxes_fn(
                concentrations[-1],
                collision_coefficients,
                evaporation_coefficients,
                wall_loss_rates,
                coagulation_sink_rates,
                sources,
                dilution_loss,
                calc_final_sources=calc_final_sources,
            )
            
            fluxes_time = time.time() - fluxes_start
            logger.debug(f"Fluxes calculated in {fluxes_time:.3f}s") 
        
        logger.debug("Creating results object...")
        results_start = time.time()

        results = SimulationResults(
            system=self.system,
            equations=self.equations,
            time=t,
            concentrations=concentrations,
            initial_conditions=initial_conditions,
            collision_coefficients=collision_coefficients,
            evaporation_coefficients=evaporation_coefficients,
            wall_loss_rates=wall_loss_rates,
            coagulation_sink_rates=coagulation_sink_rates,
            formation_rates=formation_rates,
            final_outflux_rates=final_outflux_rates,
            net_fluxes=all_net_fluxes,
            final_sources=final_sources,
        )

        results_time = time.time() - results_start
        logger.debug(f"Results object created in {results_time:.3f}s")

        total_steady_time = time.time() - steady_start
        logger.info(f"Steady state calculation completed in {total_steady_time:.3f}s")

        return results