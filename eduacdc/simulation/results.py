"""
Simulation results and analysis for ACDC.
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.axes import Axes

from ..core.equations import ClusterEquations, FluxDirection
from ..core.system import SimulationSystem
from ..utils import ureg

logger = logging.getLogger(__name__)


class SimulationResults:
    """Container for ACDC simulation results."""

    def __init__(
        self,
        system: SimulationSystem,
        equations: ClusterEquations,
        time: np.ndarray,
        concentrations: np.ndarray,
        initial_conditions: Dict[str, Dict[str, str | float]],
        collision_coefficients: np.ndarray,
        evaporation_coefficients: np.ndarray,
        wall_loss_rates: np.ndarray,
        coagulation_sink_rates: np.ndarray,
        formation_rates: Optional[np.ndarray] = None,
        final_outflux_rates: Optional[np.ndarray] = None,
        net_fluxes: Optional[OrderedDict[FluxDirection, np.ndarray]] = None,
        final_sources: Optional[np.ndarray] = None,
    ):
        self.system: SimulationSystem = system
        self.equations: ClusterEquations = equations
        self.time: np.ndarray = time
        self.concentrations: np.ndarray = concentrations  # Shape: (n_time_points, n_clusters)
        self.initial_conditions: Dict[str, Dict[str, str | float]] = initial_conditions
        self.collision_coefficients: np.ndarray = collision_coefficients
        self.evaporation_coefficients: np.ndarray = evaporation_coefficients
        self.wall_loss_rates: np.ndarray = wall_loss_rates
        self.coagulation_sink_rates: np.ndarray = coagulation_sink_rates
        self.net_fluxes: Optional[OrderedDict[FluxDirection, np.ndarray]] = net_fluxes
        self.final_sources: Optional[np.ndarray] = final_sources
        self.formation_rates: np.ndarray = formation_rates
        n_clusters = self.system.n_clusters
        self.final_outflux_rates: np.ndarray = (
            final_outflux_rates
            if final_outflux_rates is not None
            else np.zeros((n_clusters, n_clusters))
        )
        # Calculate derived quantities
        self._calculate_total_concentrations()

    def _calculate_total_concentrations(self) -> None:
        """Calculate total concentrations by cluster type."""
        n_time_points = len(self.time)

        # Initialize arrays
        self.total_neutral = np.zeros(n_time_points)
        self.total_positive = np.zeros(n_time_points)
        self.total_negative = np.zeros(n_time_points)

        for t_idx in range(n_time_points):
            for i, cluster in enumerate(self.system.clusters):
                concentration = self.concentrations[t_idx, i]

                if cluster.type.value == "neutral":
                    self.total_neutral[t_idx] += concentration
                elif cluster.type.value == "positive":
                    self.total_positive[t_idx] += concentration
                elif cluster.type.value == "negative":
                    self.total_negative[t_idx] += concentration

    
    def get_final_concentrations(self, output_units: Optional[str] = None) -> np.ndarray:
        """Get final cluster concentrations.
        Args:
            output_units (str): The units to return the concentrations in. Dimensionality: [length]**-3.
            If None, returns the concentrations in m^-3.
        Returns:
            np.ndarray: The final cluster concentrations in m^-3 by default, or in the specified units if provided.
        """
        final_concentrations = self.concentrations[-1, :]
        if output_units is None:
            return final_concentrations
        else:
            return (final_concentrations * ureg("m^-3")).to(output_units).magnitude

    def get_steady_state_concentrations(self, tolerance: float = 1e-6) -> np.ndarray:
        """Get steady-state concentrations by averaging over the last portion of the simulation."""
        # Find where concentrations have stabilized
        n_time_points = len(self.time)
        for i in range(n_time_points - 1, 0, -1):
            if not np.allclose(
                self.concentrations[i, :], self.concentrations[i - 1, :], rtol=tolerance
            ):
                break

        # Average over the stable portion
        stable_concentrations = self.concentrations[i:, :]
        return np.mean(stable_concentrations, axis=0)

    def plot_final_concentrations(self, output_units: str = "cm^-3", figsize: Tuple[int, int] = (10, 6)) -> Axes:
        """Plot final cluster concentrations.
        Args:
            output_units (str): The units to return the concentrations in. Dimensionality: [length]**-3.
            If None, returns the concentrations in cm^-3.
            figsize (tuple): The figure size.
        Returns:
            Axes: The axes of the plot.
        """
        from ..analysis.visualization import plot_final_concentrations

        ax = plot_final_concentrations(self, output_units=output_units, figsize=figsize)
        return ax

    def plot_concentrations(
        self,
        cluster_indices: Optional[List[int]] = None,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        output_units: str = "cm^-3",
    ) -> Axes:
        """Plot cluster concentrations over time."""
        from ..analysis.visualization import plot_concentrations

        ax = plot_concentrations(
            self, cluster_indices=cluster_indices, log_scale=log_scale, figsize=figsize, output_units=output_units
        )
        return ax

    def plot_total_concentrations(
        self, log_scale: bool = True, figsize: Tuple[int, int] = (10, 6)
    ) -> Axes:
        """Plot total concentrations by cluster type."""
        from ..analysis.visualization import plot_total_concentrations

        ax = plot_total_concentrations(self, log_scale=log_scale, figsize=figsize)
        return ax

    def plot_cluster_size_distribution(
        self, time_index: int = -1, figsize: Tuple[int, int] = (10, 6)
    ) -> Axes:
        """Plot cluster size distribution at a specific time."""
        from ..analysis.visualization import plot_cluster_size_distribution

        ax = plot_cluster_size_distribution(
            self, time_index=time_index, figsize=figsize
        )
        return ax

    def print_summary(self) -> None:
        """Print a summary of the simulation results."""
        print("Simulation Results Summary")
        print("=========================")
        print(f"Time span: {self.time[0]:.1f} - {self.time[-1]:.1f} s")
        print(f"Number of time points: {len(self.time)}")
        print(f"Number of clusters: {self.system.n_clusters}")

        # Formation rate summary
        final_formation_rate = self.get_final_formation_rate(output_units="cm^-3 s^-1")
        print(f"Final formation rate: {final_formation_rate:.2e} cm^-3 s^-1")

        # Final concentrations
        final_concentrations = self.get_final_concentrations(output_units="cm^-3")
        print("\nFinal Concentrations (top 5):")
        sorted_indices = np.argsort(final_concentrations)[::-1]
        for i in range(min(5, len(sorted_indices))):
            cluster_idx = sorted_indices[i]
            cluster = self.system.clusters[cluster_idx]
            concentration = final_concentrations[cluster_idx]
            print(f"  {str(cluster)}: {concentration:.2e} cm^-3")

    def save_results(self, directory: str, filename_suffix: str="results", overwrite: bool=False) -> None:
        """Save results to a file."""
        dir_path = Path(directory)
        system_config = self.system.to_config()
        equations_config = self.equations.to_config()
        config = {
            "system": system_config,
            "equations": equations_config,
            "initial_conditions": self.initial_conditions,
        }
        yaml_filename = f"{filename_suffix}.yaml"
        with open(dir_path.joinpath(yaml_filename), "w") as f:
            yaml.dump(config, f)
        numpy_filename = f"{filename_suffix}.npz"
        if not overwrite and dir_path.joinpath(numpy_filename).exists():
            raise FileExistsError(f"File {numpy_filename} already exists in {directory}")
        savez_kwargs: Dict[str, Any] = {
            "time": self.time,
            "concentrations": self.concentrations,
            "collision_coefficients": self.collision_coefficients,
            "evaporation_coefficients": self.evaporation_coefficients,
            "wall_loss_rates": self.wall_loss_rates,
            "coagulation_sink_rates": self.coagulation_sink_rates,
            "formation_rates": self.formation_rates,
            "final_outflux_rates": self.final_outflux_rates,
        }
        if self.net_fluxes is not None:
            savez_kwargs["net_fluxes"] = self.net_fluxes
        if self.final_sources is not None:
            savez_kwargs["final_sources"] = self.final_sources
        np.savez(dir_path.joinpath(numpy_filename), **savez_kwargs)

    @classmethod
    def load_from_yaml_npz(cls, yaml_filename: str, npz_filename: str) -> "SimulationResults":
        """Load the yaml config and npz file."""
        with open(yaml_filename, "r") as f:
            config = yaml.safe_load(f)
        system = SimulationSystem.from_config(config["system"])
        equations = ClusterEquations.from_config(config["equations"], system.clusters)
        initial_conditions = config["initial_conditions"]
        return cls.load_results(npz_filename, system, equations, initial_conditions)

    @classmethod
    def load_results(
        cls,
        npz_filename: str,
        system: SimulationSystem,
        equations: ClusterEquations,
        initial_conditions: Dict[str, Dict[str, str | float]],
    ) -> "SimulationResults":
        """Load results from a file."""
        data = np.load(npz_filename, allow_pickle=True)

        return cls(
            system=system,
            equations=equations,
            time=data["time"],
            concentrations=data["concentrations"],
            initial_conditions=initial_conditions,
            collision_coefficients=data["collision_coefficients"],
            evaporation_coefficients=data["evaporation_coefficients"],
            wall_loss_rates=data["wall_loss_rates"],
            coagulation_sink_rates=data["coagulation_sink_rates"],
            formation_rates=data["formation_rates"],
            final_outflux_rates=data["final_outflux_rates"],
            net_fluxes=data["net_fluxes"] if "net_fluxes" in data else None,
            final_sources=data["final_sources"] if "final_sources" in data else None,
        )


    def plot_formation_rates(
        self,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        filter_negative: bool = False,
        output_units: str = "cm^-3 s^-1",
    ) -> Axes:
        """
        Plot formation rates over time.

        Parameters
        ----------
        log_scale : bool
            Whether to use log-log scale for axes.
        figsize : tuple
            Figure size.
        filter_negative : bool
            If True, negative outgrowth rates are set to NaN (not plotted).
        output_units : str
            Units of the outgrowth rates. Default is "cm^-3 s^-1".
        """
        from ..analysis.visualization import plot_formation_rates

        return plot_formation_rates(
            self,
            log_scale=log_scale,
            figsize=figsize,
            filter_negative=filter_negative,
            output_units=output_units,
        )
    
    def get_formation_rates(self, output_units: str = "m^-3 s^-1") -> np.ndarray:
        """Get the formation rates.
        Args:
            output_units (str): The units to return the formation rates in. Dimensionality: [length]**-3 [time]**-1.
            If None, returns the formation rates in m^-3 s^-1.
        Returns:
            np.ndarray: The formation rates in the specified units.
        """
        return (self.formation_rates * ureg("m^-3 s^-1")).to(output_units).magnitude

    def get_final_formation_rate(self, output_units: str = "m^-3 s^-1") -> float:
        """Get the final formation rate.
        Args:
            output_units (str): The units to return the formation rate in. Dimensionality: [length]**-3 [time]**-1.
            If None, returns the formation rate in m^-3 s^-1.
        Returns:
            float: The final formation rate in the specified units.
        """
        return (self.formation_rates[-1] * ureg("m^-3 s^-1")).to(output_units).magnitude

    
    def plot_act_deltag_surface(self, ax: Optional[Axes] = None, output_units: str = "cm^-3") -> Axes:
        """Plot the activation energy surface."""
        from ..analysis.rates_and_deltags import plot_act_deltag_surface
        neutral_monomers = list(self.system.molecules.get_neutral_molecules().keys())
        if len(neutral_monomers) != 2:
            raise ValueError("This visualization only supports 2-component systems")
        compound_A = neutral_monomers[0]
        compound_B = neutral_monomers[1]
        max_A = self.system.clusters.get_max_molecule_count(compound_A)
        max_B = self.system.clusters.get_max_molecule_count(compound_B)
        monomer_concentrations ={}
        for i,molecule in enumerate(neutral_monomers):
            cluster_label = f"1{molecule}"
            value = self.initial_conditions[cluster_label]["value"]
            monomer_concentrations[molecule] = value
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        logger.debug("Finding activation energy surface with monomer concentrations: %s", monomer_concentrations)
        return plot_act_deltag_surface(self.system, compound_A, compound_B, max_A, max_B, monomer_concentrations, ax=ax)