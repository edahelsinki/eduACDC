"""
Process coefficient definitions and calculations for ACDC.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from ..utils.constants import BOLTZMANN_CONSTANT, PI
from .cluster_properties import ClusterProperties
from .clusters import Cluster, ClusterCollection

logger = logging.getLogger(__name__)

RECOMBINATION_COEFFICIENT = 1.6e-12  # m^3/s for ion-ion recombination
CONSTANT_ION_ENHANCEMENT_FACTOR = 10.0  # Constant ion enhancement factor


@dataclass
class ProcessCoefficientsConfiguration:
    """Configuration for process coefficients."""

    ion_collision_method: str = "su82"
    disable_evaporations: bool = False

    def __post_init__(self):
        """Post-initialization logic."""
        self.ion_collision_method = self.ion_collision_method.lower()
        if self.ion_collision_method not in ["su82", "constant"]:
            raise ValueError(
                f"Invalid ion collision method: {self.ion_collision_method}. Must be 'su82' or 'constant'."
            )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ProcessCoefficientsConfiguration":
        """Create a configuration from a configuration."""
        return cls(
            ion_collision_method=config.get("ion_collision_method", "su82"),
            disable_evaporations=config.get("disable_evaporations", False),
        )

    def to_config(self) -> Dict[str, Any]:
        """Convert configuration to a configuration."""
        return {
            "ion_collision_method": self.ion_collision_method,
            "disable_evaporations": self.disable_evaporations,
        }


class ProcessCoefficients:
    """Manages the calculation and storage of process coefficients."""

    def __init__(
        self,
        clusters: ClusterCollection,
        cluster_properties: ClusterProperties,
        configuration: ProcessCoefficientsConfiguration,
    ):
        self.clusters: ClusterCollection = clusters
        self.cluster_properties: ClusterProperties = cluster_properties
        self.configuration: ProcessCoefficientsConfiguration = configuration
        #
        self._collision_coefficients: Optional[np.ndarray] = None
        self._evaporation_coefficients: Optional[np.ndarray] = None
        self._evaporation_processes: Dict[int, Set[Tuple[int, int]]] = {}  # k-> (i, j)
        self._hard_sphere_collision_coefficients: Dict[tuple[str, str], float] = {}
        self._collision_enhancement_factors: Dict[tuple[str, str], float] = {}

    def calculate_all_coefficients(
        self,
        temperature: float,
        partial_pressure_water: float,
    ) -> None:
        """Build the complete reaction network."""
        # Calculate all coefficients
        self._collision_coefficients = self._calculate_effective_collision_coefficients(
            temperature, partial_pressure_water
        )
        # if evaporation is disabled, no need to calculate evaporation rates
        if self.configuration.disable_evaporations:
            logger.info("Evaporation rates disabled, skipping calculation.")
            self._evaporation_coefficients = np.array([])
            self._evaporation_processes = {}
        else:
            self._evaporation_coefficients, self._evaporation_processes = (
                self._calculate_effective_evaporation_coefficients(
                    temperature, partial_pressure_water
                )
            )

    def get_collision_coefficients(self) -> np.ndarray:
        """Get the collision coefficients."""
        if self._collision_coefficients is None:
            raise ValueError("Collision coefficients have not been calculated.")
        return self._collision_coefficients

    def get_evaporation_coefficients(self) -> np.ndarray:
        """Get the evaporation coefficients."""
        if self._evaporation_coefficients is None:
            raise ValueError("Evaporation coefficients have not been calculated.")
        return self._evaporation_coefficients

    def get_evaporation_processes(self) -> Dict[int, Set[Tuple[int, int]]]:
        """Get the evaporation processes."""
        if self._evaporation_processes is None:
            raise ValueError("Evaporation processes have not been calculated.")
        return self._evaporation_processes

    def _calculate_effective_collision_coefficients(
        self,
        temperature: float,
        partial_pressure_water: float,
    ) -> np.ndarray:
        """Calculate effective collision coefficients between all cluster pairs."""
        n_clusters = len(self.clusters)
        rates = np.zeros((n_clusters, n_clusters))

        for i, cluster_i in enumerate(self.clusters):
            for j, cluster_j in enumerate(self.clusters):
                # Only calculate collision rates between clusters with the same charge
                if i <= j and self.clusters.can_collide(cluster_i, cluster_j):
                    rate = self._effective_collision_rate(
                        cluster_i,
                        cluster_j,
                        temperature,
                        partial_pressure_water,
                    )
                    rates[i, j] = rate
                    rates[j, i] = rate
        return rates

    def _calculate_collision_enhancement_factor(
        self,
        c1_label: str,
        c1_mass: float,
        c1_charge: int,
        c2_label: str,
        c2_mass: float,
        c2_charge: int,
        temperature: float,
        hard_sphere_collision_rate: float,
    ) -> float:
        """Calculate the collision enhancement factor between two clusters using Su and Chesnavich, 1982

        Args:
            c1_label (str): first cluster label
            c1_mass (float): first cluster mass
            c1_charge (int): first cluster charge
            c2_label (str): second cluster label
            c2_mass (float): second cluster mass
            c2_charge (int): second cluster charge
            temperature (float): temperature
            hard_sphere_collision_rate (float): hard sphere collision rate

        Returns:
            float: collision enhancement factor
        """
        # if both are neutral, return 1.0
        if c1_charge == 0 and c2_charge == 0:
            return 1.0
        # if the ion collision method is constant, return the constant ion enhancement factor
        if self.configuration.ion_collision_method == "constant":
            return CONSTANT_ION_ENHANCEMENT_FACTOR
        # find which is neutral
        neutral_cluster = c1_label if c1_charge == 0 else c2_label

        neutral_cluster_dipole_moment = self.cluster_properties.get_dipole_moment(neutral_cluster)
        if neutral_cluster_dipole_moment is None:
            raise ValueError(f"No dipole moment for cluster {neutral_cluster}")
        neutral_cluster_polarizability = self.cluster_properties.get_polarizability(neutral_cluster)
        if neutral_cluster_polarizability is None:
            raise ValueError(f"No polarizability for cluster {neutral_cluster}")

        # calculate the collision enhancement factor
        fcr = (
            100.0
            * neutral_cluster_dipole_moment
            / math.sqrt(
                2.0 * neutral_cluster_polarizability * BOLTZMANN_CONSTANT * temperature * 1.0e23
            )
        )
        if fcr < 2.0:
            fcr = (fcr + 0.5090) ** 2 / 10.526 + 0.9754
        else:
            fcr = 0.4767 * fcr + 0.62
        fcr *= (
            2.0
            * PI
            * 4.8032e-16
            * math.sqrt(neutral_cluster_polarizability * (1.0 / c1_mass + 1.0 / c2_mass) / 1.0e27)
        )
        fcr /= hard_sphere_collision_rate
        if fcr < 1.0:
            fcr = 1.0
        return fcr

    def _hard_sphere_collision_rate(
        self,
        c1_mass: float,
        c1_radius: float,
        c2_mass: float,
        c2_radius: float,
        temperature: float,
    ) -> float:
        reduced_mass = (c1_mass * c2_mass) / (c1_mass + c2_mass)
        collision_diameter = c1_radius + c2_radius

        # Thermal velocity
        thermal_velocity = math.sqrt(8 * BOLTZMANN_CONSTANT * temperature / (PI * reduced_mass))

        # Collision cross section
        collision_cross_section = PI * collision_diameter**2

        # Collision rate
        rate = collision_cross_section * thermal_velocity
        return rate

    def _effective_collision_rate(
        self,
        cluster_i: Cluster,
        cluster_j: Cluster,
        temperature: float,
        partial_pressure_water: float,
    ) -> float:
        """Calculate effective collision rate between two clusters."""

        effective_collision_rate = 0.0
        for (
            n_waters_i,
            weight_i,
            label_i,
        ) in self.cluster_properties.iter_hydration_states(
            cluster_i, temperature, partial_pressure_water
        ):
            mass_i, radius_i = cluster_i.get_hydrated_mass_and_radius(n_waters_i)

            for (
                n_waters_j,
                weight_j,
                label_j,
            ) in self.cluster_properties.iter_hydration_states(
                cluster_j, temperature, partial_pressure_water
            ):
                # if clusters with opposite charges, use the recombination coefficient
                if cluster_i.charge * cluster_j.charge < 0:
                    hard_sphere_collision_rate = RECOMBINATION_COEFFICIENT
                    fcr = 1
                else:
                    mass_j, radius_j = cluster_j.get_hydrated_mass_and_radius(n_waters_j)

                    hard_sphere_collision_rate = self._hard_sphere_collision_rate(
                        mass_i, radius_i, mass_j, radius_j, temperature
                    )
                    fcr = self._calculate_collision_enhancement_factor(
                        label_i,
                        mass_i,
                        cluster_i.charge,
                        label_j,
                        mass_j,
                        cluster_j.charge,
                        temperature,
                        hard_sphere_collision_rate,
                    )
                self._hard_sphere_collision_coefficients[(label_i, label_j)] = (
                    hard_sphere_collision_rate
                )
                self._collision_enhancement_factors[(label_i, label_j)] = fcr
                effective_collision_rate += weight_i * weight_j * hard_sphere_collision_rate * fcr
        return effective_collision_rate

    def _evaporation_rate(
        self,
        g_k: float,
        g_i: float,
        g_j: float,
        temperature: float,
        reference_pressure: float,
        collision_rate: float,
    ) -> float:
        """Calculate the evaporation rate for an evaporation process between two clusters and a larger cluster."""
        delta_g = g_k - g_i - g_j
        energy_term = delta_g / (BOLTZMANN_CONSTANT * temperature)
        pressure_term = math.log(reference_pressure / (BOLTZMANN_CONSTANT * temperature))
        exponential_term = math.exp(energy_term + pressure_term)
        return collision_rate * exponential_term

    def _calculate_effective_evaporation_coefficients(
        self,
        temperature: float,
        partial_pressure_water: float,
    ) -> Tuple[np.ndarray, Dict[int, Set[Tuple[int, int]]]]:
        """Calculate effective evaporation coefficients for all possible evaporation processes."""
        n_clusters = len(self.clusters)
        rates = np.zeros((n_clusters, n_clusters))
        evaporation_processes: Dict[int, Set[Tuple[int, int]]] = {}  # k-> (i, j)

        for i, cluster_i in enumerate(self.clusters):
            for j, cluster_j in enumerate(self.clusters[i:], start=i):
                larger_cluster = self.clusters.get_possible_evaporated_cluster(cluster_i, cluster_j)
                if larger_cluster is not None:
                    k_idx = self.clusters.get_index_by_label(larger_cluster.label)
                    assert k_idx is not None
                    # if the reaction has already been calculated, skip
                    if k_idx in evaporation_processes and (i, j) in evaporation_processes[k_idx]:
                        continue

                    # Use the same formula as Perl ACDC:
                    # E_k,ij = K_ij * exp((ΔG / kT) + log(P / kT))
                    # where ΔG = E_k - E_i - E_j
                    # Effective evaporation rate over hydration states

                    i_hydration_distribution = self.cluster_properties.get_hydration_distribution(
                        cluster_i.label, temperature, partial_pressure_water
                    )
                    j_hydration_distribution = self.cluster_properties.get_hydration_distribution(
                        cluster_j.label, temperature, partial_pressure_water
                    )
                    k_hydration_distribution = self.cluster_properties.get_hydration_distribution(
                        larger_cluster.label, temperature, partial_pressure_water
                    )
                    # initialize the effective evaporation rate to 0
                    effective_evaporation_rate = 0.0
                    # iterate over all hydration states of k
                    for n_waters_k, weight_k in k_hydration_distribution.items():
                        label_k = self.cluster_properties.label_for(
                            larger_cluster.label, n_waters_k
                        )
                        # iterate over all hydration states of i
                        for n_waters_i, weight_i in i_hydration_distribution.items():
                            label_i = self.cluster_properties.label_for(cluster_i.label, n_waters_i)
                            # Find the hydration state of j that is compatible with the hydration state of k
                            n_waters_j = n_waters_k - n_waters_i
                            if n_waters_j < 0 or n_waters_j not in j_hydration_distribution.keys():
                                continue
                            label_j = self.cluster_properties.label_for(cluster_j.label, n_waters_j)
                            g_i = self.cluster_properties.get_reference_gibbs_free_energy(
                                label_i, temperature
                            )
                            g_j = self.cluster_properties.get_reference_gibbs_free_energy(
                                label_j, temperature
                            )
                            g_k = self.cluster_properties.get_reference_gibbs_free_energy(
                                label_k, temperature
                            )
                            hard_sphere_collision_rate = self._hard_sphere_collision_coefficients[
                                (label_i, label_j)
                            ]
                            collision_enhancement_factor = self._collision_enhancement_factors[
                                (label_i, label_j)
                            ]
                            collision_rate = (
                                hard_sphere_collision_rate * collision_enhancement_factor
                            )
                            evap_rate = self._evaporation_rate(
                                g_k,
                                g_i,
                                g_j,
                                temperature,
                                self.cluster_properties.reference_pressure,
                                collision_rate,
                            )
                            if i == j:
                                evap_rate *= 0.5
                            # get effective evaporation rate over hydration states of k
                            effective_evaporation_rate += weight_k * evap_rate

                    # Store in the better indexing: E[k,i] and E[k,j] for k -> i + j
                    rates[k_idx, i] = effective_evaporation_rate
                    rates[k_idx, j] = effective_evaporation_rate
                    # add the reaction to the dictionary
                    if k_idx not in evaporation_processes:
                        evaporation_processes[k_idx] = set()
                    # ensure the reaction is stored in the correct order
                    evaporation_processes[k_idx].add((min(i, j), max(i, j)))
        return rates, evaporation_processes
