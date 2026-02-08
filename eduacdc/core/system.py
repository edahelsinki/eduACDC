"""
ACDC system representation and management.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

from ..core.cluster_properties import ClusterProperties, EnergyData
from ..core.clusters import ClusterCollection, GenericIon
from ..core.molecules import MoleculeCollection
from ..core.process_coefficients import ProcessCoefficients, ProcessCoefficientsConfiguration
from ..utils.constants import parse_quantity

logger = logging.getLogger(__name__)


@dataclass
class AmbientConditions:
    """Physical conditions for the ambient environment."""

    temperature: float  # K
    relative_humidity: float = 0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AmbientConditions":
        """Create system conditions from a configuration."""
        if config.get("temperature") is not None:
            temperature = parse_quantity(config["temperature"], default_unit="K", target_unit="K")
        else:
            raise ValueError("Temperature is required")
        instance = cls(temperature=temperature)
        if relative_humidity:=config.get("relative_humidity"):
            if isinstance(relative_humidity, (float, int)):
                instance.relative_humidity = relative_humidity
            else:
                raise ValueError(f"Relative humidity must be a number, got {type(relative_humidity)}: {relative_humidity}")
        return instance

    @classmethod
    def from_cluster_properties(cls, cluster_properties: ClusterProperties) -> "AmbientConditions":
        """Create ambient conditions from cluster properties.

        Uses the reference temperature of the cluster properties as the ambient temperature.
        """
        return cls(temperature=cluster_properties.reference_temperature)
    
    def to_config(self) -> Dict[str, Any]:
        """Convert system conditions to a configuration."""
        return {
            "temperature": f"{self.temperature} K",
            "relative_humidity": self.relative_humidity,
        }

    def __post_init__(self):
        """Validate conditions."""
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
        if not 0 <= self.relative_humidity <= 1:
            raise ValueError(
                f"Relative humidity must be between 0 and 1, got {self.relative_humidity}"
            )
        # # Saturation vapor pressure, needed for obtaining the absolute partial pressure when RH is given as input
        self.saturation_vapor_pressure = self._get_saturation_vapor_pressure(self.temperature)
        self.partial_pressure_water = self.relative_humidity * self.saturation_vapor_pressure

    @staticmethod
    def _get_saturation_vapor_pressure(temperature: float) -> float:
        """Calculate the saturation vapor pressure for the given temperature.

        Reference: Arnold Wexler: 'Vapor Pressure Formulation for Water in Range 0 to 100 °C. A Revision'. JOURNAL OF RESEARCH of the National Bureau of Standards, 80A, 5-6, 775-785, 1976
        Args:
            temperature (float): Temperature in K

        Returns:
            float: Saturation vapor pressure in Pa
        """
        return math.exp(
            -2991.2729 * temperature ** (-2)
            - 6017.0128 / temperature
            + 18.87643854
            - 0.028354721 * temperature
            + 0.17838301e-4 * temperature**2
            - 0.84150417e-9 * temperature**3
            + 0.44412543e-12 * temperature**4
            + 2.858487 * math.log(temperature)
        )  # Pa

    def update_from_config(self, config: Dict[str, Any]) -> None:
        """Update the system conditions from a configuration."""
        updated = False
        if "temperature" in config:
            prev = self.temperature
            self.temperature = parse_quantity(config["temperature"], default_unit="K", target_unit="K")
            logger.debug(f"Updated condition: temperature from {prev} to {self.temperature}")
            updated = True
        if "relative_humidity" in config:
            prev = self.relative_humidity
            self.relative_humidity = config["relative_humidity"]
            logger.debug(f"Updated condition: relative_humidity from {prev} to {self.relative_humidity}")
            updated = True
        if updated:
            self.__post_init__()
        


class SimulationSystem:
    """
    Complete simulation system representation.

    Parameters
    ----------
    molecules : MoleculeCollection
        Collection of molecules in the system.
    clusters : ClusterCollection
        Collection of clusters in the system.
    cluster_properties : ClusterProperties
        Properties of the clusters in the system.
    process_coefficients_configuration : ProcessCoefficientsConfiguration, optional
        Configuration for the process coefficients. If not provided, uses the default configuration.
    conditions: AmbientConditions, optional
        Physical conditions for the ambient environment. If not provided, uses the reference temperature of the cluster properties.

    """

    def __init__(
        self,
        molecules: "MoleculeCollection",
        clusters: "ClusterCollection",
        cluster_properties: "ClusterProperties",
        process_coefficients_configuration: Optional["ProcessCoefficientsConfiguration"] = None,
        conditions: Optional["AmbientConditions"] = None,
    ):
        # validation
        if molecules is not clusters.molecules:
            raise ValueError("Molecules and clusters must be the same collection")

        self._molecules = molecules
        self._clusters = clusters
        self._cluster_properties = cluster_properties
        if process_coefficients_configuration is None:
            process_coefficients_configuration = ProcessCoefficientsConfiguration.from_config({})
        self._process_coefficients_configuration = process_coefficients_configuration
        if conditions is None:
            conditions = AmbientConditions.from_cluster_properties(cluster_properties)
        self._conditions = conditions
        self.validate_energy_data()
        # Build reaction network
        self.process_coefficients = ProcessCoefficients(clusters, self.cluster_properties, self.process_coefficients_configuration)
        self.calculate_all_coefficients()

    @property
    def molecules(self) -> MoleculeCollection:
        """Get the molecule collection."""
        return self._clusters.molecules

    @property
    def n_molecules(self) -> int:
        """Get the number of molecules."""
        return len(self._molecules)

    @property
    def clusters(self) -> ClusterCollection:
        """Get the cluster collection."""
        return self._clusters

    @property
    def n_clusters(self) -> int:
        """Get the number of clusters."""
        return len(self._clusters)

    @property
    def conditions(self) -> AmbientConditions:
        """Get the system conditions."""
        return self._conditions

    @property
    def cluster_properties(self) -> ClusterProperties:
        """Get the cluster properties."""
        return self._cluster_properties

    @property
    def process_coefficients_configuration(self) -> ProcessCoefficientsConfiguration:
        """Get the process coefficients configuration."""
        return self._process_coefficients_configuration

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SimulationSystem":
        """Create a system from a full configuration."""
        molecules = MoleculeCollection.from_config(config["molecules"])
        clusters = ClusterCollection.from_config(
            config["clusters"], molecules=molecules
        )
        process_coefficients_configuration = ProcessCoefficientsConfiguration.from_config(
            config.get("process_coefficients_configuration", {})
        )
        cluster_properties_config = config.get("cluster_properties")
        conditions_config = config.get("conditions")
        # Handle case if cluster_properties is not provided
        if cluster_properties_config is None:
            # try and get temperature from conditions
            if conditions_config is not None and conditions_config.get("temperature") is not None:
                default_temperature = parse_quantity(
                    conditions_config["temperature"], default_unit="K", target_unit="K"
                )
            else:
                raise ValueError(
                    "Either 'conditions.temperature' or 'cluster_properties.reference_temperature' must be provided"
                )
            # use default reference pressure of 1 atm
            default_cluster_properties_config = {
                "reference_temperature": default_temperature,
                "reference_pressure": "101325 Pa",
            }
            logger.warning(
                f"No cluster properties provided, using temperature= {default_temperature} K and pressure=1 atm"
            )
            cluster_properties = ClusterProperties.from_config(
                default_cluster_properties_config
            )
        else:
            cluster_properties = ClusterProperties.from_config(
                cluster_properties_config
            )
        
        # if conditions are provided with temperature, use that, otherwise use the default conditions
        if conditions_config is not None:
            if conditions_config.get("temperature") is not None:
                conditions = AmbientConditions.from_config(conditions_config)
            else:
                # take reference temperature from cluster properties
                _conditions_config = conditions_config.copy()
                _conditions_config["temperature"] = cluster_properties.reference_temperature
                conditions = AmbientConditions.from_config(_conditions_config)
        else:
            conditions = AmbientConditions.from_cluster_properties(cluster_properties)
        
        return cls(
            molecules,
            clusters,
            cluster_properties,
            process_coefficients_configuration,
            conditions,
        )

    def to_config(self) -> Dict[str, Any]:
        """Convert system to a configuration."""
        return {
            "molecules": self.molecules.to_config(),
            "clusters": self.clusters.to_config(),
            "conditions": self.conditions.to_config(),
            "cluster_properties": self.cluster_properties.to_config(),
            "process_coefficients_configuration": self.process_coefficients_configuration.to_config(),
        }

    def validate_energy_data(self) -> None:
            """Validate that the energy data is consistent with the cluster properties."""
            # if evaporation is disabled, no need to validate energy data
            if self.process_coefficients_configuration.disable_evaporations:
                return
            monomers = self.clusters.get_monomers()
            # set monomer entlapy and entropy to 0 kcal/mol and 0 cal/(mol*K) if not set
            for monomer in monomers:
                if not self.cluster_properties.has_energy_data(monomer.label):
                    self.cluster_properties.set_energy_for_cluster(
                        monomer.label,
                        EnergyData(
                            enthalpy=0.0,
                            entropy=0.0,
                            reference_temperature=self.cluster_properties.reference_temperature,
                        ),
                    )
            # check that all clusters have energy data
            for cluster in self.clusters:
                # generic ions can have no energy data
                if not self.cluster_properties.has_energy_data(cluster.label) and not isinstance(cluster, GenericIon):
                    raise ValueError(f"No energy data for cluster {cluster.label}")                    
    
    def calculate_all_coefficients(self) -> None:
        """Calculate all the process coefficients."""
        self.process_coefficients.calculate_all_coefficients(
            temperature=self.conditions.temperature,
            partial_pressure_water=self.conditions.partial_pressure_water,
        )

    def get_hydration_distribution(self, cluster_label: str) -> Dict[int, float]:
        """Get hydration distribution for a cluster."""
        return self.cluster_properties.get_hydration_distribution(
            cluster_label,
            self.conditions.temperature,
            self.conditions.partial_pressure_water,
        )

    def get_collision_coefficients(self) -> np.ndarray:
        """Get collision coefficients matrix."""
        return self.process_coefficients.get_collision_coefficients()

    def get_evaporation_coefficients(self) -> np.ndarray:
        """Get evaporation coefficients matrix."""
        return self.process_coefficients.get_evaporation_coefficients()

    def get_evaporation_processes(self) -> Dict[int, Set[Tuple[int, int]]]:
        """Get evaporation processes dictionary. The key is the index of the cluster that evaporates,
        and the value is a set of tuples of the indices of the clusters that are formed.

        Returns
        -------
        Dict[int, Set[Tuple[int, int]]]
            Dictionary of evaporation processes.
        """
        return self.process_coefficients.get_evaporation_processes()

    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the system."""
        return {
            "n_molecules": self.n_molecules,
            "n_clusters": self.n_clusters,
            "n_neutral_clusters": len(self.clusters.get_neutral_clusters()),
            "n_positive_clusters": len(self.clusters.get_positive_clusters()),
            "n_negative_clusters": len(self.clusters.get_negative_clusters()),
            "ambient temperature": self.conditions.temperature,
            "relative_humidity": self.conditions.relative_humidity,
            "ion_collision_method": self.process_coefficients_configuration.ion_collision_method,
            "disable_evaporations": self.process_coefficients_configuration.disable_evaporations,
        }

    def print_summary(self) -> None:
        """Print a summary of the system."""
        summary = self.get_system_summary()

        print("Simulation System Summary")
        print("==================")
        print(f"Molecules: {summary['n_molecules']}")
        print(f"Total clusters: {summary['n_clusters']}")
        print(f"  Neutral: {summary['n_neutral_clusters']}")
        print(f"  Positive: {summary['n_positive_clusters']}")
        print(f"  Negative: {summary['n_negative_clusters']}")
        print(f"Ambient temperature: {summary['ambient temperature']:.1f} K")
        print(f"Relative humidity: {summary['relative_humidity']:.2f}")
        print(f"Ion collision method: {summary['ion_collision_method']}")
        print(f"Disable evaporations: {summary['disable_evaporations']}")

    def update_conditions(self, **kwargs) -> None:
        """Update system conditions and rebuild network."""
        for key, value in kwargs.items():
            if hasattr(self.conditions, key):
                logger.debug(f"Updating {key} to {value}")
                setattr(self.conditions, key, value)
                self.conditions.__post_init__()
            else:
                raise ValueError(f"Unknown condition: {key}")

        logger.debug("Recalculating all coefficients with new conditions")
        self.calculate_all_coefficients()
    
    def update_cluster_properties(self, cluster_properties_config: Dict[str, Any]) -> None:
        """Update the cluster properties."""
        self._cluster_properties.update_from_config(cluster_properties_config)
        self.validate_energy_data()
        logger.debug("Recalculating all coefficients with new cluster properties")
        self.calculate_all_coefficients()

    def update_cluster_properties_and_conditions(self, cluster_properties_config: Dict[str, Any], conditions_config: Dict[str, Any]) -> None:
        """Update the cluster properties and conditions."""
        self._cluster_properties.update_from_config(cluster_properties_config)
        self.conditions.update_from_config(conditions_config)
        logger.debug("Recalculating all coefficients with new cluster properties and conditions")
        self.calculate_all_coefficients()

    def get_cluster_info(self, cluster_index: int) -> Dict[str, Any]:
        """Get detailed information about a specific cluster."""
        if cluster_index < 0 or cluster_index >= len(self.clusters):
            raise ValueError(f"Invalid cluster index: {cluster_index}")

        cluster = self.clusters[cluster_index]

        return {
            "label": cluster.label,
            "index": cluster_index,
            "composition": cluster.composition,
            "charge": cluster.charge,
            "type": cluster.type.value,
            "mass": cluster.mass,
            "radius": cluster.radius,
            "total_molecules": cluster.total_molecules,
            "string_representation": str(cluster),
        }

    def plot_reference_deltag_surface(
        self,
        compound_A: str,
        compound_B: str,
        max_A: int,
        max_B: int,
        charge: int = 0,
        **kwargs,
    ):
        """
        Convenience method to plot reference DeltaG surface.

        This is a thin wrapper around the analysis function.
        See acdc.analysis.rates_and_deltags.plot_reference_deltag_surface for details.
        """
        from ..analysis.rates_and_deltags import plot_reference_deltag_surface

        return plot_reference_deltag_surface(
            system=self,
            compound_A=compound_A,
            compound_B=compound_B,
            max_A=max_A,
            max_B=max_B,
            charge=charge,
            **kwargs,
        )
