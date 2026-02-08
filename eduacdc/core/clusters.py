"""
Cluster definitions and properties for ACDC.
"""

import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, overload

import numpy as np

from ..utils import PI, pint, ureg
from .molecules import MoleculeCollection, MoleculeType


class ClusterType(Enum):
    """Types of clusters."""

    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"

# Type alias for cluster or generic ion
type ClusterOrGenericIon = Cluster|GenericIon
type Composition = OrderedDict[str, int] | Dict[str, int]


def merge_compositions(comp1: Composition, comp2: Composition) -> Composition:
    """Merge two compositions."""
    merged = comp1.copy()
    for symbol, count in comp2.items():
        merged[symbol] = merged.get(symbol, 0) + count
    return merged


@dataclass
class GenericIon:
    """Represents a generic ion (positive or negative) for ionic systems."""
    
    label: str  # "neg" or "pos"
    charge: int  # -1 for negative, +1 for positive
    mass: float  #  kg/particle
    density: float  #  kg/m^3
    
    def __post_init__(self):
        """Validate and set properties."""
        if self.label not in ["neg", "pos"]:
            raise ValueError(f"Invalid generic ion label: {self.label}")
        if self.charge not in [-1, 1]:
            raise ValueError(f"Invalid charge for generic ion: {self.charge}")
        if (self.label == "neg" and self.charge != -1) or (self.label == "pos" and self.charge != 1):
            raise ValueError(f"Label {self.label} doesn't match charge {self.charge}")
    
    @property
    def type(self) -> ClusterType:
        """Get cluster type."""
        return ClusterType.NEGATIVE if self.charge == -1 else ClusterType.POSITIVE
    
    @property
    def radius(self) -> float:
        """Cluster radius in m."""
        return (3 * (self.mass / self.density) / (4 * PI)) ** (1 / 3)
    
    @property
    def diameter(self) -> float:
        """Cluster diameter in m."""
        return 2 * self.radius
    
    @property
    def composition(self) -> Composition:
        # the composition is always empty
        return OrderedDict({})
    
    @property
    def total_molecules(self) -> int:
        """Generic ions have no molecular composition."""
        return 0
    
    def get_molecule_count(self, *args, **kwargs) -> int:
        return 0
    
    def is_monomer(self) -> bool:
        """Generic ions are always considered monomers."""
        return True
    
    def get_total_molecules(self) -> int:
        """Generic ions have no molecular composition."""
        return 0

    def get_hydrated_mass_and_radius(self, n_waters: int) -> Tuple[float, float]:
        """Get hydrated mass and radius of the generic ion."""
        if n_waters == 0:
            return self.mass, self.radius
        else:
            raise ValueError(f"Cannot get hydrated mass and radius of a generic ion with {n_waters} waters")
    
    def __str__(self) -> str:
        return f"{self.label}"
    
    def __repr__(self) -> str:
        return f"{self.label}"

# The generic negative ion is a O_{2}^- ion
DEFAULT_NEG_MASS = 32.00 * ureg("g/mol").to("kg/particle").magnitude
DEFAULT_NEG_DENSITY = 1141.0 * ureg("kg/m^3").magnitude

# The generic positive ion is a H_{3}O^+ ion
DEFAULT_POS_MASS = 19.02 * ureg("g/mol").to("kg/particle").magnitude
DEFAULT_POS_DENSITY = 997.0 * ureg("kg/m^3").magnitude

GENERIC_NEG_ION = GenericIon(label="neg", charge=-1, mass=DEFAULT_NEG_MASS, density=DEFAULT_NEG_DENSITY)
GENERIC_POS_ION = GenericIon(label="pos", charge=1, mass=DEFAULT_POS_MASS, density=DEFAULT_POS_DENSITY)


WATER_MASS: float = 18.02 * ureg("g/mol").to("kg/particle").magnitude # kg/particle
WATER_DENSITY: float = 997.0 * ureg("kg/m^3").magnitude # kg/m^3
WATER_VOLUME: float = WATER_MASS / WATER_DENSITY 
WATER_RADIUS: float = (3 * WATER_VOLUME / (4 * PI)) ** (1 / 3) # m

@dataclass
class Cluster:
    """Represents a cluster in the ACDC system."""

    molecules: MoleculeCollection # the molecules in the system
    composition: Composition  # molecule_symbol -> count
    charge: int

    def __post_init__(self):
        """Allow initialization with zero mass/volume/radius; validate later."""
        self._validate_charge()

        self.mass, self.volume, self.radius = self.calculate_mass_and_volume_and_radius()
        self.total_molecules = self.get_total_molecules()
        if self.charge == 0:
            self.type = ClusterType.NEUTRAL
        elif self.charge > 0:
            self.type = ClusterType.POSITIVE
        else:
            self.type = ClusterType.NEGATIVE
        self.__hydrated_properties_cache: Dict[int, Tuple[float, float]] = {}
    
    def get_hydrated_mass_and_radius(self, n_waters: int) -> Tuple[float, float]:
        """Get hydrated mass and radius of the cluster."""
        if n_waters == 0:
            return self.mass, self.radius
        if n_waters in self.__hydrated_properties_cache:
            return self.__hydrated_properties_cache[n_waters]
        hydrated_mass: float = self.mass + n_waters * WATER_MASS
        hydrated_volume: float = self.volume + n_waters * WATER_VOLUME
        hydrated_radius: float = (3 * hydrated_volume / (4 * PI)) ** (1 / 3)
        self.__hydrated_properties_cache[n_waters] = (hydrated_mass, hydrated_radius)
        return hydrated_mass, hydrated_radius
    
    def _validate_charge(self) -> None:
        """Validate the charge of the cluster."""
        if self.charge not in [-1, 0, 1]:
            raise ValueError(f"Invalid charge {self.charge}, must be -1, 0, or 1")
        _charge = 0
        for symbol, count in self.composition.items():
            molecule = self.molecules.get_by_symbol(symbol)
            _charge += count * molecule.charge
        if _charge != self.charge:
            raise ValueError(f"Invalid charge {self.charge}, must be {_charge}")

    def calculate_mass_and_volume_and_radius(self) -> tuple[float, float, float]:
        """Calculate mass, volume, and radius of the cluster."""
        mass = 0.0
        volume = 0.0
        for symbol, count in self.composition.items():
            molecule = self.molecules.get_by_symbol(symbol)
            # Don't take the proton or the missing proton for the volume calculation
            mass += count * molecule.mass
            if molecule.type not in [MoleculeType.PROTON, MoleculeType.MISSING_PROTON]:
                volume += count * molecule.volume

        radius = (3 * volume / (4 * PI)) ** (1 / 3)
        assert radius > 0, "Radius is zero or negative"
        return mass, volume, radius

    def get_total_molecules(self) -> int:
        """Total number of molecules in the cluster."""
        num_molecules = 0
        for molecule in self.molecules:
            if molecule.type in [MoleculeType.PROTON, MoleculeType.MISSING_PROTON]:
                continue
            num_molecules += self.get_molecule_count(molecule.symbol)
        return num_molecules
    
    @property
    def diameter(self) -> float:
        """Cluster diameter in m."""
        return 2 * self.radius

    def get_molecule_count(self, symbol: str, include_ions:bool=False) -> int:
        """Get count of a specific molecule type."""
        # if include_ions is False just count the requested molecule
        count = self.composition.get(symbol, 0)
        if include_ions:
            # count the corresponding negative ion
            if (negative_ion_symbol := self.molecules.get_corresponding_negative_ion(symbol)) is not None:
                count += self.composition.get(negative_ion_symbol, 0)
            # count the corresponding positive ion
            if (positive_ion_symbol := self.molecules.get_corresponding_positive_ion(symbol)) is not None:
                # keep default 0 if the corresponding positive ion is not in the molecule collection
                # For example, if molecule is "N", the corresponding positive ion can be "1N1P" which is not in the molecule collection
                count += self.composition.get(positive_ion_symbol, 0)
        return count

    def has_molecule(self, symbol: str) -> bool:
        """Check if cluster contains a specific molecule type."""
        return symbol in self.composition and self.composition[symbol] > 0

    @property
    def label(self) -> str:
        """Create cluster label for energy lookup (without charge sign)."""
        # Match Perl create_label logic: only include molecules with count > 0, no charge
        comp_str = ""
        for symbol, count in self.composition.items():
            if count > 0:
                comp_str += f"{count}{symbol}"
        return comp_str

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return self.label

    def is_monomer(self) -> bool:
        """Return True if this cluster is a monomer (only one molecule, count == 1)."""
        return self.total_molecules == 1

    def get_complete_composition(
        self
    ) -> Composition:
        """Get composition with all molecules, including zeros."""
        complete = OrderedDict()
        for symbol in self.molecules._molecule_order:
            complete[symbol] = self.composition.get(symbol, 0)
        return complete

    def get_mass_with_units(self) -> pint.Quantity:
        """Get mass of the cluster in the output unit."""
        return (self.mass * ureg("kg/particle"))
    
    def get_volume_with_units(self) -> pint.Quantity:
        """Get volume of the cluster in the output unit."""
        return (self.volume * ureg("m^3"))
    
    def get_radius_with_units(self) -> pint.Quantity:
        """Get radius of the cluster in the output unit."""
        return (self.radius * ureg("m"))
    
    def try_charge_transformation(self, target_charge: int) -> Optional[Composition]:
        """Try to transform the cluster to a target charge.

        Args:
            target_charge (int): The desired charge of the cluster (-1, 0, or 1).

        Raises:
            ValueError: If the target charge is not -1, 0, or 1.

        Returns:
            Optional[Composition]: The transformed composition if successful, None otherwise.
        """
        if target_charge not in [-1, 0, 1]:
            raise ValueError(f"Invalid target charge: {target_charge}")
        if self.charge == target_charge:
            return self.composition.copy()
        
        composition = self.get_complete_composition()

        if target_charge == 0: # Discharge to neutral
            if self.charge == 1: # Positive -> Neutral
                # try to remove one proton
                if composition.get("P", 0) > 0:
                    composition["P"] -= 1
                    return composition
            elif self.charge == -1: # Negative -> Neutral
                # Find negative ion and convert to neutral molecule
                for symbol, count in composition.items():
                    if count > 0:
                        mol = self.molecules.get_by_symbol(symbol)
                        if mol.charge < 0:
                            neutral_mol_symbol = self.molecules.get_corresponding_neutral(mol)
                            if neutral_mol_symbol is not None:
                                composition[symbol] -= 1
                                composition[neutral_mol_symbol] += 1
                                return composition
        
        elif target_charge == 1: # charge to positive
            if self.charge == 0: # Neutral -> Positive
                # Find neutral molecule that can accept proton
                for symbol, count in composition.items():
                    if count > 0:
                        mol = self.molecules.get_by_symbol(symbol)
                        if mol.charge == 0 and self.molecules.get_corresponding_positive_ion(mol) is not None:
                            composition["P"] +=1
                            return composition
                
        elif target_charge == -1: # charge to negative
            if self.charge == 0: # Neutral -> Negative
                # Find neutral molecule that has a corresponding negative ion
                for symbol, count in composition.items():
                    if count > 0:
                        mol = self.molecules.get_by_symbol(symbol)
                        if (
                            mol.charge == 0
                            and (
                                neg_mol_symbol
                                := self.molecules.get_corresponding_negative_ion(mol)
                            )
                            is not None
                        ):
                            composition[symbol] -= 1 # remove one neutral molecule
                            composition[neg_mol_symbol] += 1 # add one negative ion
                            return composition
        # otherwise, it is not possible to transform the cluster to the target charge
        return None


class ClusterCollection:
    """Collection of clusters with lookup capabilities."""

    def __init__(
        self,
        cluster_specs: Dict[str, List[Dict]],
        molecules: MoleculeCollection,
        include_generic_neg: bool = False,
        include_generic_pos: bool = False,
    ):
        self.cluster_specs = cluster_specs
        self.molecules = molecules
        self.include_generic_neg = include_generic_neg
        self.include_generic_pos = include_generic_pos
        self.generic_ions = []
        self.clusters = self._generate_clusters(cluster_specs)
        if self.include_generic_neg:
            self.generic_ions.append(GENERIC_NEG_ION)
        if self.include_generic_pos:
            self.generic_ions.append(GENERIC_POS_ION)
        self._build_lookups()

    def _build_lookups(self) -> None:
        self._by_label = {str(cluster): cluster for cluster in self.clusters}
        self._by_charge_type = {
            ClusterType.NEUTRAL: [
                c for c in self.clusters if c.type == ClusterType.NEUTRAL
            ],
            ClusterType.POSITIVE: [
                c for c in self.clusters if c.type == ClusterType.POSITIVE
            ],
            ClusterType.NEGATIVE: [
                c for c in self.clusters if c.type == ClusterType.NEGATIVE
            ],
        }
        # By complete composition
        self._by_composition = {}
        self._index_by_composition = {}
        self._index_by_label = {}
        for i, cluster in enumerate(self.clusters):
            complete_comp = self._get_complete_composition(cluster.composition)
            key = tuple(complete_comp.items())
            self._by_composition[key] = cluster
            self._index_by_composition[key] = i
            self._index_by_label[cluster.label] = i
        
        for i, ion in enumerate(self.generic_ions):
            self._by_label[str(ion)] = ion
            self._by_charge_type[ion.type].append(ion)
            self._index_by_label[ion.label] = i + len(self.clusters)
        
        # Calculate max molecules by charge
        self.max_molecules_by_charge = {0: {}, 1: {}, -1: {}}
        for cluster in self.clusters:
            for mol, count in cluster.composition.items():
                if count > 0:
                    self.max_molecules_by_charge[cluster.charge][mol] = max(
                        self.max_molecules_by_charge[cluster.charge].get(mol, 0), count
                    )

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], molecules: MoleculeCollection
    ) -> "ClusterCollection":
        """Create a cluster collection from a configuration."""
        cluster_specs = config["specifications"]
        options = config.get("options", {})
        include_generic_neg = options.get("include_generic_neg", False)
        include_generic_pos = options.get("include_generic_pos", False)
        return cls(cluster_specs, molecules, include_generic_neg, include_generic_pos)

    def to_config(self) -> Dict[str, Any]:
        """Convert the cluster collection to a configuration."""
        return self.cluster_specs

    def _generate_clusters(self, cluster_specs: Dict[str, List[Dict]]) -> List[Cluster]:
        """Generate all clusters from specifications."""
        clusters = []
        for charge_type, specs in cluster_specs.items():
            charge = {"neutral": 0, "positive": 1, "negative": -1}[charge_type]
            for spec in specs:
                new_clusters = self._generate_clusters_from_spec(spec, charge)
                clusters.extend(new_clusters)

        return clusters

    def _generate_clusters_from_spec(self, spec: Dict, charge: int) -> List[Cluster]:
        """Generate clusters from a single specification."""
        clusters = []
        molecule_ranges = OrderedDict()
        for molecule_symbol in self.molecules._molecule_order:
            if molecule_symbol in spec.keys():
                range_spec = spec[molecule_symbol]
                if isinstance(range_spec, list) or isinstance(range_spec, tuple):
                    if len(range_spec) == 1:
                        molecule_ranges[molecule_symbol] = (
                            range_spec[0],
                            range_spec[0],
                        )
                    elif len(range_spec) == 2:
                        molecule_ranges[molecule_symbol] = (
                            range_spec[0],
                            range_spec[1],
                        )
                elif isinstance(range_spec, int):
                    molecule_ranges[molecule_symbol] = (range_spec, range_spec)
                else:
                        raise ValueError(
                            f"Invalid range specification for {molecule_symbol}: {range_spec}"
                        )

        # Generate all combinations
        from itertools import product

        symbols = list(molecule_ranges.keys())
        ranges = [
            range(molecule_ranges[sym][0], molecule_ranges[sym][1] + 1)
            for sym in symbols
        ]

        # create clusters
        for counts in product(*ranges):
            composition = {symbol: count for symbol, count in zip(symbols, counts)}
            
            # Skip empty clusters
            if sum(composition.values()) == 0:
                continue
            # Create cluster
            cluster = Cluster(
                molecules=self.molecules,
                composition=composition,
                charge=charge,
            )

            clusters.append(cluster)

        return clusters

    def _get_complete_composition(
        self, composition: Composition
    ) -> Composition:
        """Get complete composition of a cluster."""
        complete_comp = OrderedDict(
            (symbol, 0) for symbol in self.molecules._molecule_order
        )
        for symbol, count in composition.items():
            complete_comp[symbol] = count
        return complete_comp

    def get_index_by_label(self, label: str) -> Optional[int]:
        """Get index of a cluster by label."""
        if label in self._index_by_label:
            return self._index_by_label[label]
        # second try: parse label to composition
        composition = self._parse_label_to_composition(label)
        return self.get_index_by_composition(composition)

    def get_index_by_composition(self, composition: Composition) -> Optional[int]:
        """Get index of a cluster by composition."""
        complete_comp = self._get_complete_composition(composition)
        key = tuple(complete_comp.items())
        return self._index_by_composition.get(key, None)

    def get_by_label(self, label: str) -> Cluster:
        """Get cluster by string representation."""
        if label in self._by_label:
            return self._by_label[label]

        # second try: parse label to composition
        composition = self._parse_label_to_composition(label)
        cluster = self.find_by_composition(composition)
        if cluster is not None:
            return cluster
        else:
            raise KeyError(f"Cluster '{label}' not found in collection")
    
    def cluster_from_label(self, label: str) -> ClusterOrGenericIon:
        """Get a cluster object from a label."""
        composition = self._parse_label_to_composition(label)
        charge = 0
        for symbol, count in composition.items():
            molecule = self.molecules.get_by_symbol(symbol)
            charge += count * molecule.charge 
        if charge not in [-1, 0, 1]:
            raise ValueError(f"Invalid charge {charge} for cluster from label {label}")
        return Cluster(self.molecules, composition, charge)

    def get_by_charge_type(self, charge_type: ClusterType) -> List[ClusterOrGenericIon]:
        """Get all clusters of a specific type."""
        return self._by_charge_type[charge_type]

    def get_neutral_clusters(self) -> List[ClusterOrGenericIon]:
        """Get all neutral clusters."""
        return self._by_charge_type[ClusterType.NEUTRAL]

    def get_positive_clusters(self) -> List[ClusterOrGenericIon]:
        """Get all positive clusters."""
        return self._by_charge_type[ClusterType.POSITIVE]

    def get_negative_clusters(self) -> List[ClusterOrGenericIon]:
        """Get all negative clusters."""
        return self._by_charge_type[ClusterType.NEGATIVE]

    def check_if_in_collection(self, composition: Composition) -> bool:
        """Check if a cluster is in the collection."""
        complete_comp = self._get_complete_composition(composition)
        key = tuple(complete_comp.items())
        if key in self._by_composition:
            return True
        return False

    def find_by_composition(self, composition: Composition) -> Optional[Cluster]:
        """Find cluster by composition and charge."""
        complete_comp = self._get_complete_composition(composition)
        key = tuple(complete_comp.items())
        return self._by_composition.get(key)

    def composition_to_label(self, composition: Composition) -> str:
        """Convert composition to label."""
        composition = self._get_complete_composition(composition)
        label = ""
        for symbol, count in composition.items():
            if count > 0:
                label += f"{count}{symbol}"
        return label

    def _parse_label_to_composition(self, label: str) -> Composition:
        """Parse label to composition using molecule order."""
        composition = OrderedDict(
            (symbol, 0) for symbol in self.molecules._molecule_order
        )
        pattern = r"(\d+)([A-Z]+)"
        matches = re.findall(pattern, label)

        for count_str, symbol in matches:
            if symbol in composition:
                composition[symbol] = int(count_str)

        return composition

    def get_monomers(self) -> List[Cluster]:
        """Get the monomers in this collection."""
        monomers = []
        for cluster in self.clusters:
            if cluster.is_monomer():
                monomers.append(cluster)
        return monomers

    def get_labels(self) -> List[str]:
        """Get all labels of the clusters."""
        return [str(cluster) for cluster in self.clusters] + [str(ion) for ion in self.generic_ions]

    def __len__(self) -> int:
        """Get total number of clusters including generic ions."""
        return len(self.clusters) + len(self.generic_ions)

    def __iter__(self):
        """Iterate over all clusters and generic ions."""
        yield from self.clusters
        yield from self.generic_ions

    @overload
    def __getitem__(self, index: int) -> ClusterOrGenericIon: ...

    @overload  
    def __getitem__(self, index: slice) -> list[ClusterOrGenericIon]: ...

    def __getitem__(self, index: int|slice) -> ClusterOrGenericIon|list[ClusterOrGenericIon]:
        if isinstance(index, slice):
            all_items = list(self.clusters) + list(self.generic_ions)
            return all_items[index]
        elif isinstance(index, int) or np.issubdtype(type(index), np.integer):            
            # convert to Python int
            index = int(index)
            if index < 0:
                index += len(self)
            if index < len(self.clusters):
                return self.clusters[index]
            elif index < len(self):
                return self.generic_ions[index - len(self.clusters)]
            else:
                raise IndexError(f"Index {index} out of range")
        else:
            raise TypeError(f"Invalid index type: {type(index)}")
    
    def __contains__(self, cluster: str|ClusterOrGenericIon) -> bool:
        """Check if a cluster label is in the collection."""
        if isinstance(cluster, Cluster) or isinstance(cluster, GenericIon):
            return cluster.label in self._by_label
        elif isinstance(cluster, str):
            return self.get_index_by_label(cluster) is not None
        else:
            raise TypeError(f"Invalid cluster label {cluster}. Expected str or ClusterOrGenericIon, got {type(cluster)}.")
    
    def get_label(self, index: int) -> str:
        """Get label of a cluster by index."""
        if index < len(self.clusters):
            return self.clusters[index].label
        elif index < len(self):
            return self.generic_ions[index - len(self.clusters)].label
        else:
            raise IndexError(f"Index {index} out of range")

    @staticmethod
    def sort_by_total_molecules(cluster1: Cluster, cluster2: Cluster) -> tuple[Cluster,Cluster]:
        """Sort two clusters by total molecules."""
        sorted_clusters = sorted([cluster1,cluster2],key=lambda x: x.total_molecules)
        return sorted_clusters[0],sorted_clusters[1]
    
    def get_max_molecule_count(self, molecule_symbol: str) -> int:
        """Get the maximum number of a molecule in the clusters."""
        return max(cluster.get_molecule_count(molecule_symbol) for cluster in self.clusters)
    
    def _handle_ionic_recombination(self, cluster1: ClusterOrGenericIon, cluster2: ClusterOrGenericIon) -> Optional[Composition]:
        """Handle recombination of two charged clusters."""
        # if clusters are not oppositely charged cannot recombine
        if cluster1.charge * cluster2.charge >= 0:
            raise ValueError(f"Cannot combine clusters with the same charge: {cluster1} and {cluster2}")
        # find which is which by charge
        neg_cluster = cluster1 if cluster1.charge < 0 else cluster2
        pos_cluster = cluster1 if cluster1.charge > 0 else cluster2
        # Try to discharge the negative cluster to neutral
        if isinstance(neg_cluster, Cluster):
            neutral_composition_neg_cluster = neg_cluster.try_charge_transformation(0)
            if neutral_composition_neg_cluster is None:
                return None
        else:
            neutral_composition_neg_cluster = self._get_complete_composition({}) # empty composition for generic ion
        
        # Try to discharge the positive cluster to neutral
        if isinstance(pos_cluster, Cluster):
            neutral_composition_pos_cluster = pos_cluster.try_charge_transformation(0)
            if neutral_composition_pos_cluster is None:
                return None
        else:
            neutral_composition_pos_cluster = self._get_complete_composition({}) # empty composition for generic ion
        
        # Combine the neutral compositions
        combined_composition = merge_compositions(neutral_composition_neg_cluster, neutral_composition_pos_cluster)
        return combined_composition
        
    
    def get_combined_composition(self, cluster1: ClusterOrGenericIon, cluster2: ClusterOrGenericIon) -> Optional[Composition]:
        """Get the combined composition of two clusters if they can be combined.

        Args:
            cluster1 (ClusterOrGenericIon): The first cluster.
            cluster2 (ClusterOrGenericIon): The second cluster.

        Returns:
            Optional[Composition]: The combined composition if successful, None otherwise.
        """
        # if two generic ions no resulting cluster
        if isinstance(cluster1, GenericIon) and isinstance(cluster2, GenericIon):
            return None
        
        # if clusters have the same charge, no resulting cluster
        if cluster1.charge * cluster2.charge > 0:
            return None
        
        # when one cluster is positive and one cluster is negative
        if cluster1.charge * cluster2.charge < 0:
            combined_composition = self._handle_ionic_recombination(cluster1, cluster2)
            if combined_composition is None:
                return None
            return combined_composition
        # when both clusters are neutral or one is ion
        if cluster1.charge * cluster2.charge == 0:
            # if one of the clusters is a generic ion, try to charge the other cluster to the charge of the generic ion
            if isinstance(cluster1, GenericIon):
                gen_ion: GenericIon = cluster1
                assert isinstance(cluster2, Cluster)
                charged_composition = cluster2.try_charge_transformation(gen_ion.charge)
                return charged_composition if charged_composition is not None else None
            elif isinstance(cluster2, GenericIon):
                gen_ion: GenericIon = cluster2
                assert isinstance(cluster1, Cluster)
                charged_composition = cluster1.try_charge_transformation(gen_ion.charge)
                return charged_composition if charged_composition is not None else None
            
            # when both clusters are regular clusters, combine the compositions
            return merge_compositions(cluster1.composition, cluster2.composition)

    def can_collide(self, cluster1: ClusterOrGenericIon, cluster2: ClusterOrGenericIon) -> bool:
        """Basic logic to check if two clusters can collide."""
        # clusters with the same charge cannot collide
        return cluster1.charge * cluster2.charge <= 0
    
    def get_possible_evaporated_cluster(self, cluster1: ClusterOrGenericIon, cluster2: ClusterOrGenericIon) -> Optional[Cluster]:
        """Try and find the cluster which can evaporate back to the two clusters.

        Args:
            cluster1 (ClusterOrGenericIon): The first cluster.
            cluster2 (ClusterOrGenericIon): The second cluster.

        Returns:
            Optional[Cluster]: The cluster which can evaporate back to the two clusters, None otherwise.
        """
        # collisions with generic ions cannot evaporate back
        if isinstance(cluster1, GenericIon) or isinstance(cluster2, GenericIon):
            return None
       
        # if the clusters have the same charge, they cannot be formed from evaporating
        if cluster1.charge * cluster2.charge > 0:
            return None
        
        # if one positive and one negative cannot be formed from evaporating
        if cluster1.charge * cluster2.charge < 0:
            return None
        
        # if one positive or negative
        if cluster1.charge * cluster2.charge == 0:
            # find the composition of the cluster which can evaporate back to the two clusters
            combined_composition = self.get_combined_composition(cluster1, cluster2)
            # if not possible to combine the clusters, return None
            if combined_composition is None:
                return None
            # try and find the cluster in the collection
            combined_cluster = self.find_by_composition(combined_composition)
            # if found, return the cluster
            if combined_cluster is not None:
                return combined_cluster
            else:
                # otherwise, return None
                return None
        return None