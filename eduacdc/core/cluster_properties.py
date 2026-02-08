import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from ..utils import BOLTZMANN_CONSTANT, parse_quantity, pint, ureg
from .clusters import Cluster, GenericIon

LABEL_COMPONENT_PATTERN = re.compile(r"(\d+)([A-Za-z]+)")

logger = logging.getLogger(__name__)
@dataclass
class EnergyData:
    """Container for cluster energy information."""
    reference_temperature: float    # K
    gibbs_free_energy: Optional[float] = None  # J/particle
    enthalpy: Optional[float] = None          # J/particle  
    entropy: Optional[float] = None           # J/(K*particle)
    
    def __post_init__(self):
        """Validate energy consistency."""
        if (self.gibbs_free_energy is not None and 
            self.enthalpy is not None and 
            self.entropy is not None):
            raise ValueError(
                "Cannot set all three: Gibbs free energy, enthalpy, and entropy. "
                "Set either Gibbs free energy OR enthalpy + entropy."
            )
    
    def get_gibbs_free_energy(self, temperature: float) -> float:
        """Get Gibbs free energy at given temperature."""
        if self.gibbs_free_energy is not None:
            if temperature == self.reference_temperature or self.gibbs_free_energy == 0.0:
                return self.gibbs_free_energy
            else:
                raise ValueError(f"Cannot get Gibbs free energy at temperature {temperature} K because reference temperature is {self.reference_temperature} K. For temperature dependent Gibbs free energy, provide the enthalpy and entropy.")
        elif self.enthalpy is not None and self.entropy is not None:
            return self.enthalpy - temperature * self.entropy
        else:
            raise ValueError("Insufficient energy data")
    
    def to_config(self) -> Union[str, List[str]]:
        """Convert to configuration format."""
        if self.gibbs_free_energy is not None:
            val = (self.gibbs_free_energy * ureg("J/particle")).to("kcal/mol").magnitude
            return f"{val} kcal/mol"
        elif self.enthalpy is not None and self.entropy is not None:
            val_enthalpy = (self.enthalpy * ureg("J/particle")).to("kcal/mol").magnitude
            val_entropy = (self.entropy * ureg("J/particle/K")).to("cal/(mol*K)").magnitude
            return [f"{val_enthalpy} kcal/mol", f"{val_entropy} cal/(mol*K)"]
        else:
            raise ValueError("No energy data available")
    
    @classmethod
    def from_config(cls, energy_spec: Union[List[float], float, str], reference_temperature: float) -> "EnergyData":
        """Create EnergyData from configuration."""
        if isinstance(energy_spec, list) and len(energy_spec) == 2:
            # Enthalpy and entropy
            enthalpy = parse_quantity(energy_spec[0], "kcal/mol", "J/particle")
            entropy = parse_quantity(energy_spec[1], "cal/(mol*K)", "J/(K*particle)")
            return cls(enthalpy=enthalpy, entropy=entropy, reference_temperature=reference_temperature)
        elif isinstance(energy_spec, (float, str)):
            # Gibbs free energy
            gibbs = parse_quantity(energy_spec, "kcal/mol", "J/particle")
            return cls(gibbs_free_energy=gibbs, reference_temperature=reference_temperature)
        else:
            raise ValueError(f"Invalid energy specification: {energy_spec}. Reference temperature: {reference_temperature}")

def compute_hydration_distribution(
    energy_grid: Dict[int, float],
    temperature: float,
    partial_pressure_water: float,
    reference_pressure: float,
) -> Dict[int, float]:
    """Compute normalized hydrate weights."""

    beta = BOLTZMANN_CONSTANT * temperature
    if beta <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    ratio = partial_pressure_water / reference_pressure
    weights: Dict[int, float] = {}
    for n_waters, delta_g in energy_grid.items():
        if delta_g is None:
            raise ValueError(f"No energy data for {n_waters} waters.")
        weights[n_waters] = ratio ** n_waters * math.exp(-delta_g / beta)

    total = sum(weights.values())
    normalized_weights = {n_waters: weight / total for n_waters, weight in weights.items()}

    sum_normalized_weights = sum(normalized_weights.values())
    if sum_normalized_weights < 0.99:
        raise ValueError(f"Hydration distribution failed to normalize (sum={sum_normalized_weights}).")

    return normalized_weights

@dataclass
class ClusterProperties:
    """Container for all cluster thermodynamic and electronic properties."""
    
    # reference temperature and pressure for energy data
    reference_temperature: float # in K
    reference_pressure: float # in Pa
    # Core properties - indexed by cluster label
    energies: Dict[str, EnergyData] = field(default_factory=dict)
    dipole_moments: Dict[str, float] = field(default_factory=dict)  # Debye
    polarizabilities: Dict[str, float] = field(default_factory=dict)  # Å³
    _label_cache: Dict[str, str] = field(default_factory=dict)
    hydrate_energies: Dict[str, Dict[int, EnergyData]] = field(default_factory=dict)
    _hydration_distribution_cache: Dict[Tuple[str, float, float], Dict[int, float]] = field(default_factory=dict)
    water_symbol: str = "W"
    disable_hydration_parsing: bool = False
    
    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize a cluster label by sorting it based on molecule"""
        matches = LABEL_COMPONENT_PATTERN.findall(label)
        matches.sort(key=lambda x: x[1])
        return "".join([f"{count}{symbol}" for count, symbol in matches])

    @staticmethod
    def _parse_label_components(label: str) -> List[Tuple[int, str]]:
        """Return a list of (count, symbol) pairs contained in the label."""
        components = LABEL_COMPONENT_PATTERN.findall(label)
        return [(int(count), symbol) for count, symbol in components]

    def get_corresponding_dry(self, cluster_label: str, water_symbol: Optional[str] = None) -> Tuple[str, int]:
        """Return the dry cluster label and hydrate count for a hydrated label."""
        if self.disable_hydration_parsing:
            return (self._normalize_label(cluster_label), 0)
        symbol = water_symbol or self.water_symbol
        components = self._parse_label_components(cluster_label)
        if not components:
            return (self._normalize_label(cluster_label), 0)

        n_waters = 0
        dry_components: List[Tuple[int, str]] = []
        for count, mol_symbol in components:
            if mol_symbol == symbol:
                n_waters += count
            else:
                dry_components.append((count, mol_symbol))

        if n_waters == 0:
            return (self._normalize_label(cluster_label), 0)

        if not dry_components:
            return ("", n_waters)

        dry_label = "".join(f"{count}{mol_symbol}" for count, mol_symbol in dry_components)
        return (self._normalize_label(dry_label), n_waters)
    
    def label_for(self, cluster_label: str, n_waters: int) -> str:
        """Return the label for a specific cluster and water count."""
        if n_waters == 0:
            return cluster_label
        return f"{cluster_label}{n_waters}{self.water_symbol}"
    
    def _set_property(self, property_dict: Dict[str, Any], cluster_label: str, value: Any):
        """Set a property for a specific cluster."""
        # Always store the normalized label
        normalized_label = self._normalize_label(cluster_label)
        property_dict[normalized_label] = value
        # cache mapping if different
        if cluster_label != normalized_label:
            self._label_cache[cluster_label] = normalized_label

    def _get_property(self, property_dict: Dict[str, Any], cluster_label: str) -> Optional[Any]:
        """Get a property for a specific cluster."""
        # Direct lookup 
        if cluster_label in property_dict:
            return property_dict[cluster_label]
        # check label cache
        if cluster_label in self._label_cache:
            return property_dict[self._label_cache[cluster_label]]
        
        # otherwise, try to normalize again
        normalized_label = self._normalize_label(cluster_label)
        if normalized_label in property_dict:
            # cache the mapping 
            self._label_cache[cluster_label] = normalized_label
            return property_dict[normalized_label]
        # otherwise, return None
        return None
    
    def get_energy_for_cluster(self, cluster_label: str) -> Optional[EnergyData]:
        """Get energy data for a specific cluster."""
        energy = self._get_property(self.energies, cluster_label)
        if energy is not None:
            return energy

        dry_label, n_waters = self.get_corresponding_dry(cluster_label)
        if n_waters == 0 or dry_label == "":
            return None
        normalized = self._normalize_label(dry_label)
        hydrate_map = self.hydrate_energies.get(normalized)
        if hydrate_map is None:
            return None
        return hydrate_map.get(n_waters)
    
    def get_dipole_moment(self, cluster_label: str) -> Optional[float]:
        """Get dipole moment for a specific cluster."""
        return self._get_property(self.dipole_moments, cluster_label)
    
    def get_polarizability(self, cluster_label: str) -> Optional[float]:
        """Get polarizability for a specific cluster."""
        return self._get_property(self.polarizabilities, cluster_label)
    
    def set_energy_for_cluster(self, cluster_label: str, energy_data: EnergyData):
        """Set energy data for a specific cluster."""
        dry_label, n_waters = self.get_corresponding_dry(cluster_label)
        if n_waters == 0:
            self._set_property(self.energies, cluster_label, energy_data)
            self._invalidate_hydration_distribution_cache(cluster_label)
            return

        if dry_label == "":
            logger.warning("Hydrate label %s does not map to a dry cluster; skipping.", cluster_label)
            return

        normalized = self._normalize_label(dry_label)
        hydrates = self.hydrate_energies.setdefault(normalized, {})
        hydrates[n_waters] = energy_data
        self._invalidate_hydration_distribution_cache(normalized)
    
    def set_dipole_moment(self, cluster_label: str, dipole_moment: float):
        """Set dipole moment for a specific cluster."""
        self._set_property(self.dipole_moments, cluster_label, dipole_moment)
    
    def set_polarizability(self, cluster_label: str, polarizability: float):
        """Set polarizability for a specific cluster."""
        self._set_property(self.polarizabilities, cluster_label, polarizability)
    
    def remove_cluster_properties(self, cluster_label: str):
        """Remove all properties for a specific cluster."""
        normalized_label = self._normalize_label(cluster_label)
        self.energies.pop(normalized_label, None)
        self.dipole_moments.pop(normalized_label, None)
        self.polarizabilities.pop(normalized_label, None)
        self.hydrate_energies.pop(normalized_label, None)
        self._invalidate_hydration_distribution_cache(normalized_label)
        
        # remove entries that point to this normalized label
        keys_to_remove = [k for k, v in self._label_cache.items() if v == normalized_label]
        for k in keys_to_remove:
            del self._label_cache[k]
    
    def has_energy_data(self, cluster_label: str) -> bool:
        """Check if a cluster has any properties defined."""
        if self._get_property(self.energies, cluster_label) is not None:
            return True
        dry_label, n_waters = self.get_corresponding_dry(cluster_label)
        if n_waters == 0 or dry_label == "":
            return False
        normalized = self._normalize_label(dry_label)
        return normalized in self.hydrate_energies and n_waters in self.hydrate_energies[normalized]

    def has_hydrate_data(self, cluster_label: str) -> bool:
        """Return True when hydrate thermodynamics exist for the cluster."""
        normalized = self._normalize_label(cluster_label)
        return normalized in self.hydrate_energies and bool(self.hydrate_energies[normalized])

    def get_hydrate_energies(self, cluster_label: str) -> Dict[int, EnergyData]:
        """Get hydrate energies indexed by water count for a cluster."""
        normalized = self._normalize_label(cluster_label)
        return self.hydrate_energies.get(normalized, {})

    def get_hydration_distribution(
        self,
        cluster_label: str,
        temperature: float,
        partial_pressure_water: float,
    ) -> Dict[int, float]:
        """Get hydration distribution for a cluster."""
        # check cache
        normalized = self._normalize_label(cluster_label)
        cache_key = (normalized, temperature, partial_pressure_water)
        if cache_key in self._hydration_distribution_cache:
            return self._hydration_distribution_cache[cache_key]
        dry_distribution = {0:1} # default for dry cluster with 0 water molecules
        # if no water vapor pressure, return the dry distribution
        if partial_pressure_water == 0:
            self._hydration_distribution_cache[cache_key] = dry_distribution
            return dry_distribution
        # start by getting the dry energy
        dry_energy = self.energies.get(normalized)
        if dry_energy is None:
            raise ValueError(f"Cannot get hydration distribution for cluster {cluster_label} because it has no dry energy data.")
        energy_grid = {0: dry_energy.get_gibbs_free_energy(temperature)}
        # then get the hydrate energies
        hydrate_map = self.hydrate_energies.get(normalized)
        # if no hydrate energies, return the dry distribution
        if hydrate_map is None:
            self._hydration_distribution_cache[cache_key] = dry_distribution
            return dry_distribution
        # otherwise, compute the hydration distribution
        else:
            for n_waters, energy in hydrate_map.items():
                energy_grid[n_waters] = energy.get_gibbs_free_energy(temperature)
            try:
                distribution = compute_hydration_distribution(energy_grid, temperature, partial_pressure_water, self.reference_pressure)
            except ValueError as e:
                logger.warning(f"Error computing hydration distribution for cluster {cluster_label}: {e}. Not using the hydrate distribution.")
                self._hydration_distribution_cache[cache_key] = dry_distribution
                return dry_distribution
            self._hydration_distribution_cache[cache_key] = distribution
            return distribution

    def iter_hydration_states(
        self,
        cluster: Cluster,
        temperature: float,
        partial_pressure_water: float,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Iterate over hydration states for a cluster."""
        # if a generic ion or no water vapor pressure, return the dry distribution
        if isinstance(cluster, GenericIon) or partial_pressure_water == 0:
            yield 0, 1, cluster.label
        else:
            distribution = self.get_hydration_distribution(
                cluster.label, temperature, partial_pressure_water
            )
            for n_waters, weight in distribution.items():
                yield n_waters, weight, self.label_for(cluster.label, n_waters)

    def validate_against_clusters(self, cluster_collection) -> bool:
        """Validate that all clusters in collection have required properties."""
        for cluster in cluster_collection:
            if not self.has_energy_data(cluster.label):
                return False
        return True
    
    def get_actual_gibbs_free_energy(self, cluster: Cluster, temperature: float, monomer_vapor_pressures: Dict[str, float]) -> float:
        """Calculate actual Gibbs free energy at given temperature and monomer vapor pressures.

        Args:
            cluster (Cluster): Cluster to calculate actual energy for
            temperature (float): Temperature in K
            monomer_vapor_pressures (Dict[str, float]): Dictionary of monomer vapor pressures in Pa

        Raises:
            ValueError: If insufficient thermodynamic data is provided

        Returns:
            float: Actual Gibbs free energy in J
        """
        energy_data = self.get_energy_for_cluster(cluster.label)
        if energy_data is None:
            raise ValueError(f"No energy data for cluster {cluster.label}")
        deltag_ref = energy_data.get_gibbs_free_energy(temperature)
        term = 0.0
        for symbol, count in cluster.composition.items():
            if count == 0:
                continue
            if symbol in monomer_vapor_pressures:
                monomer_vapor_pressure = monomer_vapor_pressures[symbol]
                term += count * math.log(monomer_vapor_pressure / self.reference_pressure)

        deltag_act = deltag_ref - BOLTZMANN_CONSTANT * temperature * term
        return deltag_act
    
    def get_reference_gibbs_free_energy(self, cluster: str, temperature: float) -> float:
        """Get reference Gibbs free energy for a specific cluster."""
        energy_data = self.get_energy_for_cluster(cluster)
        if energy_data is None:
            raise ValueError(f"No energy data for cluster {cluster}")
        try:
            return energy_data.get_gibbs_free_energy(temperature)
        except ValueError as e:
            raise ValueError(f"Error getting reference Gibbs free energy for cluster {cluster}: {e}")
    
    def get_reference_gibbs_free_energy_with_units(self, cluster: Cluster, temperature: float) -> pint.Quantity:
        """Get reference Gibbs free energy for a specific cluster with units."""
        energy_data = self.get_energy_for_cluster(cluster.label)
        if energy_data is None:
            raise ValueError(f"No energy data for cluster {cluster.label}")
        return energy_data.get_gibbs_free_energy(temperature) * ureg("J/particle")

    def to_config(self) -> Dict[str, Any]:
        """Convert to configuration format."""
        config = {}
        
        # Convert energies (dry + hydrate entries)
        energy_entries: Dict[str, Union[str, List[str]]] = {}
        if self.energies:
            energy_entries.update({
                label: energy.to_config()
                for label, energy in self.energies.items()
            })

        if self.hydrate_energies:
            for label, hydrates in self.hydrate_energies.items():
                for n_waters, energy in hydrates.items():
                    hydrate_label = f"{label}{n_waters}{self.water_symbol}"
                    energy_entries[hydrate_label] = energy.to_config()

        if energy_entries:
            config["energies"] = energy_entries
        
        # Convert other properties
        if self.dipole_moments:
            config["dipole_moments"] = {
                label: f"{value} Debye" 
                for label, value in self.dipole_moments.items()
            }
        
        if self.polarizabilities:
            config["polarizabilities"] = {
                label: f"{value} Å³" 
                for label, value in self.polarizabilities.items()
            }
        
        if self.disable_hydration_parsing:
            config["disable_hydration_parsing"] = True
        
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ClusterProperties":
        """Create ClusterProperties from configuration."""
        if "reference_temperature" in config:
            reference_temperature = parse_quantity(config["reference_temperature"], "K", "K")
        else:
            raise ValueError("Reference temperature is required")
        if "reference_pressure" in config:
            reference_pressure = parse_quantity(config["reference_pressure"], "Pa", "Pa")
        else:
            raise ValueError("Reference pressure is required")
        instance = cls(
            reference_temperature=reference_temperature,
            reference_pressure=reference_pressure,
        )
        if "disable_hydration_parsing" in config:
            instance.disable_hydration_parsing = config["disable_hydration_parsing"]
        if "energies" in config:
            for label, energy_spec in config["energies"].items():
                instance.set_energy_for_cluster(label, EnergyData.from_config(energy_spec, instance.reference_temperature))
        
        if "dipole_moments" in config:
            for label, dipole_str in config["dipole_moments"].items():
                instance.set_dipole_moment(label, parse_quantity(dipole_str, "debye", "debye"))
        
        if "polarizabilities" in config:
            for label, polar_str in config["polarizabilities"].items():
                instance.set_polarizability(label, parse_quantity(polar_str, "angstrom**3", "angstrom**3"))
        return instance

    def update_from_config(self, config: Dict[str, Any]) -> None:
        """Update the cluster properties from a configuration."""
        if "reference_temperature" in config:
            logger.debug("Updating reference temperature from configuration")
            self.reference_temperature = parse_quantity(config["reference_temperature"], "K", "K")
        if "reference_pressure" in config:
            logger.debug("Updating reference pressure from configuration")
            self.reference_pressure = parse_quantity(config["reference_pressure"], "Pa", "Pa")
            # The reference pressure affects the hydration distribution, so clear the cache
            self._hydration_distribution_cache = {}
        
        if "energies" in config:
            logger.debug("Updating energies from configuration")
            for label, energy_spec in config["energies"].items():
                self.set_energy_for_cluster(label, EnergyData.from_config(energy_spec, self.reference_temperature))
        if "dipole_moments" in config:
            logger.debug("Updating dipole moments from configuration")
            for label, dipole_str in config["dipole_moments"].items():
                self.set_dipole_moment(label, parse_quantity(dipole_str, "Debye", "Debye"))
        if "polarizabilities" in config:
            logger.debug("Updating polarizabilities from configuration")
            for label, polar_str in config["polarizabilities"].items():
                self.set_polarizability(label, parse_quantity(polar_str, "Å³", "Å³"))

    def copy(self) -> "ClusterProperties":
        """Create a deep copy of the properties."""
        return ClusterProperties(
            reference_temperature=self.reference_temperature,
            reference_pressure=self.reference_pressure,
            energies=self.energies.copy(),
            dipole_moments=self.dipole_moments.copy(),
            polarizabilities=self.polarizabilities.copy(),
            _label_cache=self._label_cache.copy(),
            hydrate_energies={label: hydrates.copy() for label, hydrates in self.hydrate_energies.items()},
            _hydration_distribution_cache=self._hydration_distribution_cache.copy(),
            water_symbol=self.water_symbol,
            disable_hydration_parsing=self.disable_hydration_parsing,
        )

    def _invalidate_hydration_distribution_cache(self, cluster_label: str) -> None:
        """Remove cached hydration states affected by the cluster label."""
        normalized = self._normalize_label(cluster_label)
        keys_to_remove = [key for key in self._hydration_distribution_cache if key[0] == normalized]
        for key in keys_to_remove:
            del self._hydration_distribution_cache[key]