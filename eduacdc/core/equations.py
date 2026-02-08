"""
Symbolic equation generation for ACDC using SymPy.
"""

import logging
import time
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import sympy as sym
from sympy import MatrixSymbol, Symbol

from eduacdc.core.clusters import ClusterCollection, GenericIon
from eduacdc.core.molecules import MoleculeCollection

logger = logging.getLogger(__name__)

# Named tuple for flux direction
FluxDirection = namedtuple("FluxDirection", ["source", "destination"])

# TypedDicts for symbolic and callable equations and fluxes
SymbolicAndCallableEquations = TypedDict("SymbolicAndCallableEquations", {
    "symbolic": sym.Expr,
    "callable": Callable
})
SymbolicAndCallableFluxes = TypedDict("SymbolicAndCallableFluxes", {
    "symbolic": Dict[FluxDirection, sym.Expr],
    "callable": Callable
})

# Named tuples for reaction indices
CollisionFormation = namedtuple("CollisionFormation", ["reactant1_idx", "reactant2_idx"])
CollisionLoss = namedtuple("CollisionLoss", ["partner_idx", "product_idx"])
EvaporationFormation = namedtuple("EvaporationFormation", ["partner_idx", "reactant_idx"])
EvaporationLoss = namedtuple("EvaporationLoss", ["product1_idx", "product2_idx"])
BoundaryClusterFormation = namedtuple("BoundaryClusterFormation", ["reactant1_idx", "reactant2_idx"])
BoundaryClusterMonomerFormation = namedtuple("BoundaryClusterMonomerFormation", ["reactant1_idx", "reactant2_idx", "monomer_count"])
BoundaryLoss = namedtuple("BoundaryLoss", ["partner_idx", "boundary_cluster_idx"])

@dataclass
class BoundaryProcess:
    """Represents a boundary process with full details."""
    reactant_indices: Tuple[int, int]
    product_label: str
    boundary_cluster_idx: int
    removed_monomers: Dict[int, int] # index of removed monomer and count
    is_used: bool  # Whether this reaction is actually used in equations


@dataclass
class OutgrowthCollision:
    """Represents an outgrowth collision."""
    reactant_indices: Tuple[int, int]
    product_composition: Dict[str, int]
    product_charge: int


class ProcessTracker:
    """
    Centralized process tracking with clear data structures for fast equation generation.
    
    This class organizes all discovered processes and provides efficient access patterns
    for building symbolic equations in ClusterEquations.
    """
    
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        
        # Organized by reaction type and direction using named tuples
        self.collision_formation_indices: Dict[int, List[CollisionFormation]] = {
            i: [] for i in range(n_clusters)
        }
        self.collision_loss_indices: Dict[int, List[CollisionLoss]] = {
            i: [] for i in range(n_clusters)
        }
        
        self.evaporation_formation_indices: Dict[int, List[EvaporationFormation]] = {
            i: [] for i in range(n_clusters)
        }
        self.evaporation_loss_indices: Dict[int, List[EvaporationLoss]] = {
            i: [] for i in range(n_clusters)
        }
        
        self.boundary_cluster_formation_indices: Dict[int, List[BoundaryClusterFormation]] = {
            i: [] for i in range(n_clusters)
        }
        self.boundary_cluster_monomer_formation_indices: Dict[int, List[BoundaryClusterMonomerFormation]] = {
            i: [] for i in range(n_clusters)
        }
        self.boundary_loss_indices: Dict[int, List[BoundaryLoss]] = {
            i: [] for i in range(n_clusters)
        }
        
        self.outgrowth_loss_indices: Dict[int, List[int]] = {
            i: [] for i in range(n_clusters)
        }

        # for collisions of generic charger ions
        self.recombination_loss_indices: Dict[int, List[int]] = {
            i: [] for i in range(n_clusters)
        }
        
        # Special process collections
        self.boundary_processes: Dict[Tuple[int, int], BoundaryProcess] = {}
        self.outgrowth_collisions: Dict[Tuple[int, int], OutgrowthCollision] = {}
        
    def add_collision(self, i: int, j: int, k: int):
        """Add collision process i + j -> k"""
        # Formation for product k: (i,j)
        formation = CollisionFormation(reactant1_idx=i, reactant2_idx=j)
        self.collision_formation_indices[k].append(formation)
        
        # Loss for reactant i: (j,k)
        loss_i = CollisionLoss(partner_idx=j, product_idx=k)
        self.collision_loss_indices[i].append(loss_i)
        
        # Loss for reactant j: (i,k) 
        if i != j:
            loss_j = CollisionLoss(partner_idx=i, product_idx=k)
            self.collision_loss_indices[j].append(loss_j)
   
    def add_evaporation(self, k: int, i: int, j: int):
        """Add evaporation process k -> i + j"""
        # Loss for reactant k: (i,j)
        loss = EvaporationLoss(product1_idx=i, product2_idx=j)
        self.evaporation_loss_indices[k].append(loss)
        
        # Formation for product i: (j,k)
        formation_i = EvaporationFormation(partner_idx=j, reactant_idx=k)
        self.evaporation_formation_indices[i].append(formation_i)
        
        # Formation for product j: (i,k)
        if i != j:
            formation_j = EvaporationFormation(partner_idx=i, reactant_idx=k)
            self.evaporation_formation_indices[j].append(formation_j)
    
    def add_boundary(self, i: int, j: int, product_label: str, boundary_cluster_idx: int, removed_monomers: Dict[int, int], is_used: bool):
        """Add boundary process i + j -> boundary_cluster + monomer_count * monomer"""
        boundary_process = BoundaryProcess(
            reactant_indices=(i, j),
            product_label=product_label,
            boundary_cluster_idx=boundary_cluster_idx,
            removed_monomers=removed_monomers,
            is_used=is_used
        )
        self.boundary_processes[(i, j)] = boundary_process
        
        # Formation for boundary cluster: (i,j)
        formation = BoundaryClusterFormation(reactant1_idx=i, reactant2_idx=j)
        self.boundary_cluster_formation_indices[boundary_cluster_idx].append(formation)

        # Formation for monomers: (i,j,count)  
        for removed_monomer_idx, removed_mol_count in removed_monomers.items():
            formation_monomer = BoundaryClusterMonomerFormation(reactant1_idx=i, reactant2_idx=j, monomer_count=removed_mol_count)
            self.boundary_cluster_monomer_formation_indices[removed_monomer_idx].append(formation_monomer)
        
        # Loss for reactant i: (j,boundary_cluster_idx)
        loss_i = BoundaryLoss(partner_idx=j, boundary_cluster_idx=boundary_cluster_idx)
        self.boundary_loss_indices[i].append(loss_i)

        # Loss for reactant j: (i,boundary_cluster_idx)
        if i != j:
            loss_j = BoundaryLoss(partner_idx=i, boundary_cluster_idx=boundary_cluster_idx)
            self.boundary_loss_indices[j].append(loss_j)
    
    def add_outgrowth(self, i: int, j: int, product_composition: Dict[str, int], product_charge: int):
        """Add outgrowth collision i + j -> outgrows"""
        outgrowth_collision = OutgrowthCollision(
            reactant_indices=(i, j),
            product_composition=product_composition,
            product_charge=product_charge
        )
        self.outgrowth_collisions[(i, j)] = outgrowth_collision

        # Loss for reactant i: (j,None)
        self.outgrowth_loss_indices[i].append(j)
        
        # Loss for reactant j: (i,None)
        if i != j:
            self.outgrowth_loss_indices[j].append(i)
        
    def add_recombination(self, i: int, j: int):
        """Add recombination process of generic ion + generic ion -> nothing"""
        self.recombination_loss_indices[i].append(j)
        if i != j:
            self.recombination_loss_indices[j].append(i)

    # Convenience methods for equation building
    def get_collision_birth_terms(self, cluster_idx: int) -> List[CollisionFormation]:
        """Get collision formation terms for building equations."""
        return self.collision_formation_indices[cluster_idx]
    
    def get_collision_death_terms(self, cluster_idx: int) -> List[CollisionLoss]:
        """Get collision loss terms for building equations."""
        return self.collision_loss_indices[cluster_idx]
    
    def get_evaporation_birth_terms(self, cluster_idx: int) -> List[EvaporationFormation]:
        """Get evaporation formation terms for building equations."""
        return self.evaporation_formation_indices[cluster_idx]
    
    def get_evaporation_death_terms(self, cluster_idx: int) -> List[EvaporationLoss]:
        """Get evaporation loss terms for building equations."""
        return self.evaporation_loss_indices[cluster_idx]
    
    def get_boundary_death_terms(self, cluster_idx: int) -> List[BoundaryLoss]:
        """Get boundary loss terms for building equations."""
        return self.boundary_loss_indices[cluster_idx]
    
    def get_boundary_cluster_birth_terms(self, cluster_idx: int) -> List[BoundaryClusterFormation]:
        """Get boundary cluster formation terms for building equations."""
        return self.boundary_cluster_formation_indices[cluster_idx]
    
    def get_boundary_cluster_monomer_birth_terms(self, cluster_idx: int) -> List[BoundaryClusterMonomerFormation]:
        """Get boundary cluster monomer formation terms for building equations."""
        return self.boundary_cluster_monomer_formation_indices[cluster_idx]
    
    def get_outgrowth_death_terms(self, cluster_idx: int) -> List[int]:
        """Get outgrowth loss terms for building equations."""
        return self.outgrowth_loss_indices[cluster_idx]
    
    def get_recombination_death_terms(self, cluster_idx: int) -> List[int]:
        """Get recombination loss terms for building equations."""
        return self.recombination_loss_indices[cluster_idx]
    
    def get_process_statistics(self) -> Dict[str, int]:
        """Get statistics about the processes."""
        stats = {
            "total_clusters": self.n_clusters,
            "total_collisions": 0,
            "total_evaporations": 0,
            "boundary_processes": sum(1 for r in self.boundary_processes.values() if r.is_used),
            "useless_boundary_processes": sum(1 for r in self.boundary_processes.values() if not r.is_used),
            "outgrowth_collisions": len(self.outgrowth_collisions),
        }

        recombination_loss_terms = 0        
        for cluster_idx in range(self.n_clusters):
            stats["total_collisions"] += len(self.get_collision_birth_terms(cluster_idx))
            stats["total_evaporations"] += len(self.get_evaporation_death_terms(cluster_idx))
            recombination_loss_terms += len(self.get_recombination_death_terms(cluster_idx))
        
        if recombination_loss_terms == 2: # there are 2 recombination reactions
            stats["total_recombinations"] = 1
        elif recombination_loss_terms == 0:
            stats["total_recombinations"] = 0
        else:
            raise ValueError(f"Recombination loss terms should be 0 or 2, but got {recombination_loss_terms}")

        return stats
    
    def _get_boundary_process(self, i: int, j: int) -> Optional[BoundaryProcess]:
        """Get the boundary process for a cluster pair."""
        if (i,j) in self.boundary_processes:
            return self.boundary_processes[(i,j)]
        elif (j,i) in self.boundary_processes:
            return self.boundary_processes[(j,i)]
        else:
            return None
    
    def _get_outgrowth_collision(self, i: int, j: int) -> Optional[OutgrowthCollision]:
        """Get the outgrowth collision for a cluster pair."""
        if (i,j) in self.outgrowth_collisions:
            return self.outgrowth_collisions[(i,j)]
        elif (j,i) in self.outgrowth_collisions:
            return self.outgrowth_collisions[(j,i)]
        else:
            return None
    
    def get_collision_loss_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the collision loss dataframe for a cluster."""
        rows = []
        for reaction in self.get_collision_death_terms(cluster_idx):
            rows.append({
                "collider_1_idx":cluster_idx,
                "collider_2_idx":reaction.partner_idx,
                "product_idx":reaction.product_idx,
            })
        return pd.DataFrame(rows)

    def get_evaporation_loss_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the evaporation loss dataframe for a cluster."""
        rows = []
        for reaction in self.get_evaporation_death_terms(cluster_idx):
            rows.append({
                "evaporator_idx":cluster_idx,
                "product_1_idx":reaction.product1_idx,
                "product_2_idx":reaction.product2_idx,
            })
        return pd.DataFrame(rows)
    
    def get_collision_formation_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the collision formation dataframe for a cluster."""
        rows = []
        for reaction in self.get_collision_birth_terms(cluster_idx):
            rows.append({
                "collider_1_idx":reaction.reactant1_idx,
                "collider_2_idx":reaction.reactant2_idx,
                "product_idx":cluster_idx,
            })
        return pd.DataFrame(rows)
    
    def get_evaporation_formation_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the evaporation formation dataframe for a cluster."""
        rows = []
        for reaction in self.get_evaporation_birth_terms(cluster_idx):
            rows.append({
                "evaporator_idx":reaction.reactant_idx,
                "product_1_idx":reaction.partner_idx,
                "product_2_idx":cluster_idx,
            })
        return pd.DataFrame(rows)
    
    def get_boundary_death_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the boundary death dataframe for a cluster."""
        rows = []
        for partner_idx, _ in self.get_boundary_death_terms(cluster_idx):
            _boundary_process = self._get_boundary_process(cluster_idx, partner_idx)
            if _boundary_process is not None:
                rows.append({
                    "reactant_1_idx":cluster_idx,
                    "reactant_2_idx":partner_idx,
                    "product_label":_boundary_process.product_label,
                    "boundary_cluster_idx":_boundary_process.boundary_cluster_idx,
                    "removed_monomers":_boundary_process.removed_monomers,
                    "is_used":_boundary_process.is_used,
                })
        return pd.DataFrame(rows)
    
    def get_boundary_cluster_formation_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the boundary cluster formation dataframe for a cluster."""
        rows = []
        for reactant1_idx, reactant2_idx in self.get_boundary_cluster_birth_terms(cluster_idx):
            _boundary_process = self._get_boundary_process(reactant1_idx, reactant2_idx)
            if _boundary_process is not None:  
                rows.append({
                    "reactant_1_idx":reactant1_idx,
                    "reactant_2_idx":reactant2_idx,
                    "product_label":_boundary_process.product_label,
                    "boundary_cluster_idx":_boundary_process.boundary_cluster_idx,
                    "removed_monomers":_boundary_process.removed_monomers,
                    "is_used":_boundary_process.is_used,
                })
        return pd.DataFrame(rows)
    
    def get_boundary_monomer_formation_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the boundary monomer formation dataframe for a cluster."""
        rows = []
        for reactant1_idx, reactant2_idx, _  in self.get_boundary_cluster_monomer_birth_terms(cluster_idx):
            _boundary_process = self._get_boundary_process(reactant1_idx, reactant2_idx)
            if _boundary_process is not None:
                rows.append({
                    "reactant_1_idx":reactant1_idx,
                    "reactant_2_idx":reactant2_idx,
                    "product_label":_boundary_process.product_label,
                    "boundary_cluster_idx":_boundary_process.boundary_cluster_idx,
                    "removed_monomers":_boundary_process.removed_monomers,
                    "is_used":_boundary_process.is_used,
                })
        return pd.DataFrame(rows)
    
    def get_outgrowth_death_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the outgrowth death dataframe for a cluster."""
        rows = []
        for partner_idx in self.get_outgrowth_death_terms(cluster_idx):
            _outgrowth_collision = self._get_outgrowth_collision(cluster_idx, partner_idx)
            if _outgrowth_collision is not None:
                rows.append({
                    "collider_1_idx":cluster_idx,
                    "collider_2_idx":partner_idx,
                    "product_composition":_outgrowth_collision.product_composition,
                    "product_charge":_outgrowth_collision.product_charge,
                })
        return pd.DataFrame(rows)
    
    
@dataclass
class EquationsConfiguration:
    """Configuration for generating the equations."""

    outgrowth_rules: Dict[str, List[Dict[str, int]]]
    use_acid_base_logic: bool = True
    disable_nonmonomers: bool = False
    keep_useless_collisions: bool = False
    disable_evaporations: bool = False

    def __post_init__(self):
        """Post-initialization logic."""
        self.charge_map = {"neutral": 0, "positive": 1, "negative": -1}
        self.outgrowth_rules_by_charge = {
            self.charge_map[charge]: rules for charge, rules in self.outgrowth_rules.items()
        }
        

    @classmethod
    def from_config(
        cls, config: Dict[str, Any]
    ) -> "EquationsConfiguration":
        """Create a configuration from a configuration."""
        outgrowth_rules = config["outgrowth_rules"]
        options = config.get("options", {})
        return cls(
            outgrowth_rules=outgrowth_rules,
            use_acid_base_logic=options.get("use_acid_base_logic", True),
            disable_nonmonomers=options.get("disable_nonmonomers", False),
            keep_useless_collisions=options.get("keep_useless_collisions", False),
            disable_evaporations=options.get("disable_evaporations", False),
        )

    def to_config(self) -> Dict[str, Any]:
        """Convert configuration to a configuration."""
        return {
            "outgrowth_rules": self.outgrowth_rules,
            "options": {
                "use_acid_base_logic": self.use_acid_base_logic,
                "disable_nonmonomers": self.disable_nonmonomers,
                "keep_useless_collisions": self.keep_useless_collisions,
                "disable_evaporations": self.disable_evaporations,
            },
        }
    
    def print_summary(self):
        """Print the summary of the equations configuration."""
        print("Equations Configuration:")
        print("=" * 50)
        print("Outgrowth rules:")
        for charge, rules in self.outgrowth_rules.items():
            print(f"  {charge}:")
            for rule in rules:
                rule_str = " and ".join([f">= {count} {cluster}" for cluster, count in rule.items()])
                print(f"    {rule_str}")
                
        print(f"Use acid base logic: {self.use_acid_base_logic}")
        print(f"Disable nonmonomers: {self.disable_nonmonomers}")
        print(f"Keep useless collisions: {self.keep_useless_collisions}")
        print(f"Disable evaporations: {self.disable_evaporations}")


class ClusterEquations:
    """
    Generate symbolic equations for ACDC system using SymPy.

    Parameters
    ----------
    clusters : ClusterCollection
        The cluster collection object containing clusters.
    equations_config : EquationsConfiguration
        The configuration object specifying boundaries, outgrowth, etc.
    """

    def __init__(self, clusters: ClusterCollection, equations_config: EquationsConfiguration):
        self.clusters: ClusterCollection = clusters
        self.molecules: MoleculeCollection = clusters.molecules
        self.equations_config: EquationsConfiguration = equations_config
        self._validate_equations_config()
        self.n_clusters = len(self.clusters)
        self.rebuild()
    
    
    def _validate_equations_config(self):
        available_molecules = set(self.molecules.get_molecule_order())
        for charge_type, rules in self.equations_config.outgrowth_rules.items():
            for rule in rules:
                cluster_charge = 0
                for molecule_symbol, count in rule.items():
                    # check if the molecule is available
                    if molecule_symbol not in available_molecules:
                        raise ValueError(f"Molecule {molecule_symbol} not found in system for outgrowth rule {rule} with charge type {charge_type}. Available molecules: {available_molecules}")
                    # check if value is a non-negative integer
                    if not isinstance(count, int) or count < 0:
                        raise ValueError(f"Invalid count {count} for molecule {molecule_symbol} for outgrowth rule {rule} with charge type {charge_type}. Count must be non-negative integer.")
                    cluster_charge += count * self.molecules.get_by_symbol(molecule_symbol).charge
                # check if the cluster charge is the same as the charge type
                if cluster_charge != self.equations_config.charge_map[charge_type]:
                    raise ValueError(f"Invalid charge {cluster_charge} for outgrowth rule {rule} with charge type {charge_type}. Expected {self.equations_config.charge_map[charge_type]} for {charge_type} outgrowth rule")

    def rebuild(self):
        """Setup the equations."""
        self.process_tracker = ProcessTracker(self.n_clusters)
        self.max_molecules_by_charge = {0: {}, 1: {}, -1: {}}
        for cluster in self.clusters:
            for mol, count in cluster.composition.items():
                if count > 0:
                    self.max_molecules_by_charge[cluster.charge][mol] = max(
                        self.max_molecules_by_charge[cluster.charge].get(mol, 0), count
                    )
        # Create symbolic variables
        self._create_symbols()

        # Build cluster concentration equations
        self.ode_equations = self._build_ode_equations()

        # Build formation rate and outflux matrix equations
        self._build_formation_rate_and_outflux()

        self._ode_function_cache = {}

    def _create_symbols(self):
        """Create symbolic variables for the system."""
        # Concentration variables [C_i]
        self.concentrations = sym.symbols(f"C_0:{self.n_clusters}")

        # Coefficient matrices
        self.collision_coefficients = MatrixSymbol("K", self.n_clusters, self.n_clusters)
        if self.equations_config.disable_evaporations:
            logger.info("Evaporation coefficients disabled, skipping creation of evaporation coefficients matrix.")
            self.evaporation_coefficients = MatrixSymbol("E", 0, 0)
        else:
            self.evaporation_coefficients = MatrixSymbol("E", self.n_clusters, self.n_clusters)

        # Source and external loss terms
        self.sources = sym.symbols(f"S_0:{self.n_clusters}")
        self.wall_losses = sym.symbols(f"L_0:{self.n_clusters}")
        self.coagulation_sinks = sym.symbols(f"CS_0:{self.n_clusters}")
        self.dilution_loss = sym.Symbol("dil")

        # Time variable
        self.time = Symbol("t")

    def _build_ode_equations(self) -> List[sym.Expr]:
        """Build the system of differential equations for cluster concentrations.

        Returns:
            List[sym.Expr]: The system of differential equations.
        """
        logger.info("Building ODE equations...")
        start_time = time.time()
        # First, build comprehensive process tracking
        self._populate_process_tracker()

        equations = []

        for i in range(self.n_clusters):
            # dC_i/dt = sources + collision_birth - collision_death + evaporation_birth - evaporation_death - losses
            _birth_terms = []
            _death_terms = []

            _birth_terms.append(self.sources[i])  # Source terms

            # Collision terms
            _birth_terms.append(self._collision_birth_terms(i))
            _death_terms.append(self._collision_death_terms(i))

            # Evaporation terms
            if not self.equations_config.disable_evaporations:
                _birth_terms.append(self._evaporation_birth_terms(i))
                _death_terms.append(self._evaporation_death_terms(i))

            # External loss terms (wall loss, coagulation sink, dilution)
            _death_terms.append(self.wall_losses[i] * self.concentrations[i])
            _death_terms.append(self.coagulation_sinks[i] * self.concentrations[i])
            _death_terms.append(self.dilution_loss * self.concentrations[i])

            # Boundary terms
            _death_terms.append(self._boundary_death_terms(i))
            _birth_terms.append(self._boundary_birth_terms(i))

            # Collisions that lead out of the system
            _death_terms.append(self._outgrowth_death_terms(i))
            
            # Recombination terms (for generic ion + generic ion reactions)
            _death_terms.append(self._recombination_death_terms(i))

            # add all the terms together and subtract the death terms from the birth terms
            equations.append(sym.Add(*_birth_terms) - sym.Add(*_death_terms))
        end_time = time.time()
        logger.info(
            f"ODE equations built in {end_time - start_time} seconds"
        )
        return equations

    def _populate_process_tracker(self):
        """Track processes for the system."""
        start_time = time.time()
        logger.debug("Tracking reactions...")
        # Loop through all possible cluster pairs
        for i in range(self.n_clusters):
            for j in range(i, self.n_clusters):
                self._process_cluster_pair(i, j)
        end_time = time.time()
        logger.debug(
            f"Reactions tracked in {end_time - start_time} seconds"
        )

    def _process_cluster_pair(self, i: int, j: int) -> None:
        """Process a pair of clusters to determine all possible reactions.
        This function is called for all pairs of clusters in the system.
        It determines all possible reactions between the two clusters.
        It also checks if the reaction leads to outgrowth, boundary conditions, or evaporation.
        It adds the reactions to the reaction tracker.

        Args:
            i (int): index of first cluster
            j (int): index of second cluster
        """
        cluster_i = self.clusters[i]
        cluster_j = self.clusters[j]

        # Check if collision is possible
        if self.can_collide(cluster_i, cluster_j):
            # if both clusters are generic ions, add recombination process
            if isinstance(cluster_i, GenericIon) and isinstance(cluster_j, GenericIon):
                self.process_tracker.add_recombination(i, j)
                return
            # Determine collision product
            combined_comp = self.clusters.get_combined_composition(cluster_i, cluster_j)
            # if the clusters cannot be combined, skip
            if combined_comp is None:
                return
            combined_charge = cluster_i.charge + cluster_j.charge
            # Check boundary conditions
            should_leave, boundary_comp, removed_mols = self._check_boundary_conditions(
                combined_comp, combined_charge
            )
            if should_leave:
                # Outgrowth - cluster leaves system
                self.process_tracker.add_outgrowth(i, j, combined_comp, combined_charge)
            elif boundary_comp is not None and removed_mols is not None:
                # Brought back to boundary
                self._process_boundary_process(
                    i, j, combined_comp, boundary_comp, removed_mols
                )
            else:
                # Normal collision within system
                k = self.clusters.get_index_by_composition(combined_comp)
                if k is not None:
                    self.process_tracker.add_collision(i, j, k)
                    # Check if reverse evaporation is possible 
                    if not self.equations_config.disable_evaporations and self.can_evaporate(
                        self.clusters[k], cluster_i, cluster_j
                    ):
                        self.process_tracker.add_evaporation(k, i, j)

    def can_collide(self, cluster1, cluster2) -> bool:
        # Disable nonmonomer collisions if configured
        if self.equations_config.disable_nonmonomers:
            if not cluster1.is_monomer() and not cluster2.is_monomer():
                return False
        
        # use the basic logic to check if two clusters can collide
        return self.clusters.can_collide(cluster1, cluster2)

    def can_evaporate(self, product, cluster1, cluster2) -> bool:
        # try and find the cluster which can evaporate back to the two clusters
        evaporated_cluster = self.clusters.get_possible_evaporated_cluster(cluster1, cluster2)
        if evaporated_cluster is None:
            return False
        # check if the evaporated cluster is the same as the product
        if evaporated_cluster.composition != product.composition or evaporated_cluster.charge != product.charge:
            return False
        return True
        

    def _process_boundary_process(
        self,
        i: int,
        j: int,
        original_comp: Dict[str, int],
        boundary_comp: Dict[str, int],
        removed_mols: Dict[str, int],
    ):
        """Process a boundary reaction (cluster brought back to system boundary)

        Args:
            i (int): reactant 1 index
            j (int): reactant 2 index
            original_comp (Dict[str, int]): original composition
            boundary_comp (Dict[str, int]): boundary cluster composition
            removed_mols (Dict[str, int]): removed monomers and their counts

        Raises:
            ValueError: if multiple molecules are removed to bring back to boundary
            ValueError: if no molecules are removed to bring back to boundary
            ValueError: if removed molecule is not found
        """
        # Find the boundary cluster
        boundary_cluster_idx = self.clusters.get_index_by_composition(boundary_comp)
        if boundary_cluster_idx is None:
            raise ValueError(
                f"Boundary cluster not found for composition: {boundary_comp}"
            )

        boundary_label = self.clusters.composition_to_label(boundary_comp)
        original_label = self.clusters.composition_to_label(original_comp)
        if len(removed_mols) == 0:
            raise ValueError(
                f"No molecules removed to bring back to boundary: {self.clusters[i]} + {self.clusters[j]} -> {original_label} -> {boundary_label} (no molecules removed)"
            )
        removed_monomers = {}
        for removed_mol, removed_mol_count in removed_mols.items():
            # get the index of the removed monomer (eg, for molecule A, find index of monomer 1A)
            removed_monomer_idx = self.clusters.get_index_by_composition({removed_mol: 1})
            if removed_monomer_idx is None:
                raise ValueError(f"Removed molecule not found: {removed_mol}")
            removed_monomers[removed_monomer_idx] = removed_mol_count

        # If this is monomer + cluster -> monomer + cluster, we can just skip it
        if not self.equations_config.keep_useless_collisions and (
            (self.clusters[i].is_monomer() and boundary_cluster_idx == j)
            or (self.clusters[j].is_monomer() and boundary_cluster_idx == i)
        ):
            is_used=False
        else:
            is_used=True

        # Add boundary process to process tracker
        self.process_tracker.add_boundary(i, j, original_label, boundary_cluster_idx, removed_monomers, is_used)


    def _collision_birth_terms(self, i: int) -> sym.Add:
        """Generate collision birth terms for cluster i using reaction tracking."""
        terms = []

        # Use the collision formation indices
        for reactant1_index, reactant2_index in self.process_tracker.get_collision_birth_terms(i):
            rate = self.collision_coefficients[reactant1_index, reactant2_index]
            if reactant1_index == reactant2_index:
                rate *= 0.5  # Avoid double counting
            terms.append(rate * self.concentrations[reactant1_index] * self.concentrations[reactant2_index])

        return sym.Add(*terms)

    def _collision_death_terms(self, i: int) -> sym.Add:
        """Generate collision death terms for cluster i using reaction tracking."""
        terms = []

        # Use the collision loss indices
        for partner_index, product_index in self.process_tracker.get_collision_death_terms(i):
            rate = self.collision_coefficients[i, partner_index]
            # if i ==j then rate will be 2*0.5 which is 1
            terms.append(rate * self.concentrations[i] * self.concentrations[partner_index])

        return sym.Add(*terms)

    def _evaporation_birth_terms(self, i: int) -> sym.Add:
        """Generate evaporation birth terms for cluster i using reaction tracking."""
        terms = []

        # Use the evaporation formation indices
        # reactant_index -> partner_index + i
        for partner_index, reactant_index in self.process_tracker.get_evaporation_birth_terms(i):
            rate = self.evaporation_coefficients[reactant_index, i]
            if i == partner_index:
                rate *= 2  # Symmetry factor for identical products
            terms.append(rate * self.concentrations[reactant_index])

        return sym.Add(*terms)

    def _evaporation_death_terms(self, i: int) -> sym.Add:
        """Generate evaporation death terms for cluster i using reaction tracking."""
        terms = []

        # Use the evaporation loss indices
        # i -> product1_index + product2_index
        for product1_index, product2_index in self.process_tracker.get_evaporation_death_terms(i):
            rate = self.evaporation_coefficients[i, product1_index]
            terms.append(rate * self.concentrations[i])

        return sym.Add(*terms)

    def _boundary_death_terms(self, i: int) -> sym.Add:
        """Generate boundary death terms for cluster i using reaction tracking."""
        terms = []

        # Use the boundary collision loss indices
        # i + partner_index -> brought back to boundary
        for partner_index, _ in self.process_tracker.get_boundary_death_terms(i):
            rate = self.collision_coefficients[i, partner_index]
            # if i == partner_index then rate will be 2*0.5 which is 1
            terms.append(rate * self.concentrations[i] * self.concentrations[partner_index])

        return sym.Add(*terms)

    def _boundary_birth_terms(self, i: int) -> sym.Add:
        """Generate boundary birth terms for cluster i using reaction tracking."""
        terms = []

        # Add cluster formation terms for removed molecules (if not monomer)
        for reactant1_index, reactant2_index in self.process_tracker.get_boundary_cluster_birth_terms(i):
            rate = self.collision_coefficients[reactant1_index, reactant2_index]
            if reactant1_index == reactant2_index:
                rate *= 0.5 # avoid double counting
            terms.append(rate * self.concentrations[reactant1_index] * self.concentrations[reactant2_index])

        # If monomer, add monomer formation terms from boundary collision
        for reactant1_index, reactant2_index, count in self.process_tracker.get_boundary_cluster_monomer_birth_terms(i):
            rate = self.collision_coefficients[reactant1_index, reactant2_index]
            if reactant1_index == reactant2_index:
                rate *= 0.5 # avoid double counting
            # multiply by count because we might be removing multiple monomers
            terms.append(count * rate * self.concentrations[reactant1_index] * self.concentrations[reactant2_index])

        return sym.Add(*terms)
    
    def _outgrowth_death_terms(self, i: int) -> sym.Add:
        """Generate outgrowth death terms for cluster i using reaction tracking."""
        terms = []
        for partner_index in self.process_tracker.get_outgrowth_death_terms(i):
            rate = self.collision_coefficients[i, partner_index]
            # if i == partner_index then rate will be 2*0.5 which is 1
            terms.append(rate * self.concentrations[i] * self.concentrations[partner_index])
        return sym.Add(*terms)
    
    def _recombination_death_terms(self, i: int) -> sym.Add:
        """Generate recombination death terms if any generic ion + generic ion -> nothing reaction is used."""
        terms = []
        for partner_index in self.process_tracker.get_recombination_death_terms(i):
            rate = self.collision_coefficients[i, partner_index]
            # if i == partner_index then rate will be 2*0.5 which is 1
            terms.append(rate * self.concentrations[i] * self.concentrations[partner_index])
        return sym.Add(*terms)

    def _can_form_cluster(self, i: int, j: int, target: int) -> bool:
        """Check if clusters i and j can form cluster target."""
        cluster_i = self.clusters[i]
        cluster_j = self.clusters[j]
        target_cluster = self.clusters[target]

        # Combine compositions
        combined = {}
        for symbol in set(cluster_i.composition.keys()) | set(
            cluster_j.composition.keys()
        ):
            count = cluster_i.get_molecule_count(symbol) + cluster_j.get_molecule_count(
                symbol
            )
            if count > 0:
                combined[symbol] = count

        # Check if combined composition matches target
        return (
            combined == target_cluster.composition
            and cluster_i.charge + cluster_j.charge == target_cluster.charge
        )

    def _check_boundary_conditions(
        self, cluster_composition: Dict[str, int], charge: int
    ) -> Tuple[bool, Optional[Dict[str, int]], Optional[Dict[str, int]]]:
        """Check if a cluster should leave the system or be brought back to boundary.

        Returns:
            (should_leave, boundary_composition, removed_molecules)
            - should_leave: True if cluster should nucleate out of system
            - boundary_composition: New composition if brought back to boundary
            - removed_molecules: Molecules removed to bring back to boundary
        """
        # Check if cluster is within system bounds
        if self.clusters.check_if_in_collection(cluster_composition):
            return False, None, None

        # Check outgrowth criteria
        if self.satisfies_outgrowth_criteria(
            cluster_composition, charge
        ):
            return True, None, None

        # Try to bring cluster back to boundary by removing molecules
        boundary_comp, removed_mols = self.bring_to_boundary(
            cluster_composition, charge
        )
        if boundary_comp is not None and self.clusters.check_if_in_collection(
            boundary_comp
        ):
            return False, boundary_comp, removed_mols

        # If we can't bring it back, it should leave
        return True, None, None

    def create_piecewise_ode_function(
        self, const_clusters: Optional[set] = None, sum_const_clusters: Optional[dict] = None
    ) -> Callable:
        """Create a piecewise ODE function of all clusters for solving the system using lambdify directly.

        Args:
            const_clusters: Set of cluster indices with constant concentrations.
            sum_const_clusters: Dictionary of cluster indices with sum of constant concentrations.
        Returns:
            ode_function: A callable function that takes in the time, concentrations, collision rates, evaporation rates, sources, wall losses, and coagulation sinks.
            The function returns the derivative of the concentrations.
        """
        ode_time = time.time()

        # Handle constant clusters
        if const_clusters is None:
            const_clusters = set()
        if sum_const_clusters is None:
            sum_const_clusters = dict()

        # Create a key for the cache that is unique for the given const_clusters and sum_const_clusters
        key = (
            frozenset(const_clusters), # clusters with constant concentrations
            frozenset(sum_const_clusters.keys()), # clusters with sum of constant concentrations    
            frozenset([frozenset(constraint_data.get("independent_clusters", [])) for constraint_data in sum_const_clusters.values()]), # independent clusters for each sum of constant clusters
        )

        # Check if the function is already cached
        if key in self._ode_function_cache:
            ode_function = self._ode_function_cache[key]
            logger.debug("ODE function was cached")
        else:
            logger.debug("ODE function was not cached. Creating ODE function.")
            # Create modified equations for constant clusters
            modified_equations = []
            for i, eq in enumerate(self.ode_equations):
                if i in const_clusters:
                    # For constant clusters, derivative is always zero
                    modified_equations.append(sym.Integer(0))
                elif i in sum_const_clusters:
                    _equation = sym.Integer(0)
                    # rate of change of constant clusters is the negative of the sum of the rates of change of the independent clusters
                    for j in sum_const_clusters[i]["independent_clusters"]:
                        _equation += self.ode_equations[j]
                    _equation *= -1
                    modified_equations.append(_equation)
                else:
                    modified_equations.append(eq)

            # Get the symbols for the system
            symbols = (
                self.time,
                self.concentrations,
                self.collision_coefficients,
                self.evaporation_coefficients,
                self.sources,
                self.wall_losses,
                self.coagulation_sinks,
                self.dilution_loss,
            )

            # Create a lambdify function for the entire system
            ode_function = sym.lambdify(
                symbols,
                modified_equations,
                modules="numpy",
                docstring_limit=0,  # This is to avoid the docstring from being printed
            )
            # Cache the function
            self._ode_function_cache[key] = ode_function
            logger.debug(
                f"ODE function created in {time.time() - ode_time:.3f} seconds"
            )
        return ode_function

    def get_ode_function(
        self,
        collision_coefficients: np.ndarray,
        evaporation_coefficients: np.ndarray,
        wall_loss_rates: np.ndarray,
        coagulation_sink_rates: np.ndarray,
        sources: np.ndarray,
        dilution_loss: float,
        const_clusters: Optional[set] = None,
        sum_const_clusters: Optional[dict] = None,
    ) -> Callable:
        """Create a callable ODE function in the form of f(t, concentrations) for numerical integration using scipy.

        Args:
            collision_coefficients: Collision rate matrix
            evaporation_coefficients: Evaporation rate matrix
            wall_loss_rates: Wall loss rates for each cluster
            coagulation_sink_rates: Coagulation sink rates for each cluster
            sources: Source terms for each cluster
            dilution_loss: Dilution loss rate
            const_clusters: Set of cluster indices with constant concentrations.
            sum_const_clusters: Dictionary of cluster indices with sum of constant concentrations.
        """
        # create the ODE function
        ode_function = self.create_piecewise_ode_function(const_clusters, sum_const_clusters)
        if sum_const_clusters is not None:
            # if there are sum of constant clusters, we need to update the concentrations to maintain the fixed concentrations
            def ode_system(t, concentrations):
                # update the concentrations with the sum of the constant clusters
                for cluster_idx, constraint_data in sum_const_clusters.items():
                    # calculate the residual of the sum of the constant clusters
                    fixed_concentration = constraint_data["value"]
                    independent_clusters = constraint_data["independent_clusters"]
                    _sum = sum(concentrations[independent_clusters])
                    residual = fixed_concentration - _sum
                    if residual < 0:
                        # scale the independent clusters to maintain the fixed concentration
                        scale_factor = fixed_concentration / _sum
                        for independent_cluster_idx in independent_clusters:
                            concentrations[independent_cluster_idx] = concentrations[independent_cluster_idx] * scale_factor
                        concentrations[cluster_idx] = 0
                    else:
                        # if the residual is positive, set the fixed concentration to the residual
                        concentrations[cluster_idx] = residual
                # call the ODE function
                return ode_function(
                    t,
                    concentrations,
                    collision_coefficients,
                    evaporation_coefficients,
                    sources,
                    wall_loss_rates,
                    coagulation_sink_rates,
                    dilution_loss,
                )
        else:
            # if there are no sum of constant clusters, just call the ODE function
            def ode_system(t, concentrations):
                return ode_function(
                    t,
                    concentrations,
                    collision_coefficients,
                    evaporation_coefficients,
                    sources,
                    wall_loss_rates,
                    coagulation_sink_rates,
                    dilution_loss,
                )

        return ode_system

    def print_ode_equation_cluster(self, cluster_idx: int):
        """Print the ODE equation for a specific cluster."""
        cluster = self.clusters[cluster_idx]
        print(f"d[{str(cluster)}]/dt = {self.ode_equations[cluster_idx]} + ...")

    def print_ode_equations(self, max_clusters: int = 5):
        """Print the equations in a readable format."""
        print("ACDC ODE Equations:")
        print("=" * 50)

        for i in range(self.n_clusters):
            self.print_ode_equation_cluster(i)

            # Limit output for large systems
            if i >= max_clusters - 1 and i < len(self.ode_equations) - 1:
                print(f"... ({len(self.ode_equations) - max_clusters} more clusters)")
                break
        print()

    def print_max_reactions_strings(self, reactions: List[str], max_terms: int = 5, padding: str = "   "):
        """Print the reactions in a readable format."""
        for reaction in reactions[:max_terms]:
            print(f"{padding}{reaction}")
        if len(reactions) > max_terms:
            print(f"{padding}... ({len(reactions) - max_terms} more reactions)")
        print()

    def get_collision_loss_reactions(self, cluster_idx: int) -> List[str]:
        """Get the collision loss reactions for a specific cluster."""
        collision_loss_indices = self.process_tracker.get_collision_death_terms(cluster_idx)
        return [f"{self.clusters[cluster_idx]} + {self.clusters[partner_index]} -> {self.clusters[product_index]}" for partner_index, product_index in collision_loss_indices]
    
    def get_evaporation_loss_reactions(self, cluster_idx: int) -> List[str]:
        """Get the evaporation loss reactions for a specific cluster."""
        evaporation_loss_indices = self.process_tracker.get_evaporation_death_terms(cluster_idx)
        return [
            f"{self.clusters[cluster_idx]} -> {self.clusters[product1_index]} + {self.clusters[product2_index]}"
            for product1_index, product2_index in evaporation_loss_indices
        ]
    
    def get_outgrowth_loss_reactions(self, cluster_idx: int) -> List[str]:
        """Get the outgrowth loss reactions for a specific cluster."""
        outgrowth_death_indices = self.process_tracker.get_outgrowth_death_terms(cluster_idx)
        reactions = []
        for partner_idx in outgrowth_death_indices:
            outgrowth_collision = self.process_tracker._get_outgrowth_collision(cluster_idx, partner_idx)
            if outgrowth_collision is not None:
                outgrowth_product_label = self.clusters.composition_to_label(outgrowth_collision.product_composition)
                reactions.append(f"{self.clusters[cluster_idx]} + {self.clusters[partner_idx]} -> {outgrowth_product_label} -> outgrows")
        return reactions
    
    def get_collision_formation_reactions(self, cluster_idx: int) -> List[str]:
        """Get the collision formation reactions for a specific cluster."""
        collision_form_indices = self.process_tracker.get_collision_birth_terms(cluster_idx)
        return [
            f"{self.clusters[reactant1_index]} + {self.clusters[reactant2_index]} -> {self.clusters[cluster_idx]}"
            for reactant1_index, reactant2_index in collision_form_indices
        ]
    
    def get_evaporation_formation_reactions(self, cluster_idx: int) -> List[str]:
        """Get the evaporation formation reactions for a specific cluster."""
        evaporation_form_indices = self.process_tracker.get_evaporation_birth_terms(cluster_idx)
        return [
            f"{self.clusters[evaporator_index]} -> {self.clusters[cluster_idx]} + {self.clusters[partner_index]}"
            for partner_index, evaporator_index in evaporation_form_indices
        ]

      
    def get_boundary_death_reactions(self, cluster_idx: int) -> List[str]:
        """Get the boundary death reactions for a specific cluster."""
        boundary_death_indices = self.process_tracker.get_boundary_death_terms(cluster_idx)
        reactions = []
        for partner_index, _ in boundary_death_indices:
                if (cluster_idx, partner_index) in self.process_tracker.boundary_processes:
                    _boundary_process = self.process_tracker.boundary_processes[(cluster_idx, partner_index)]
                else:
                    _boundary_process = self.process_tracker.boundary_processes[(partner_index, cluster_idx)]
                if _boundary_process.is_used:
                    _reaction_str = f"{self.clusters[cluster_idx]} + {self.clusters[partner_index]} -> {_boundary_process.product_label} -> {self.clusters[_boundary_process.boundary_cluster_idx]}"
                    for removed_monomer_idx, removed_mol_count in _boundary_process.removed_monomers.items():
                        _reaction_str += f" + {self.clusters[removed_monomer_idx]} (x{removed_mol_count})"
                    reactions.append(_reaction_str)
        return reactions

    def get_boundary_formation_reactions(self, cluster_idx: int) -> List[str]:
        """Get the boundary formation reactions for a specific cluster."""
        boundary_birth_indices = self.process_tracker.get_boundary_cluster_birth_terms(cluster_idx)
        reactions = []
        if boundary_birth_indices:
            for reactant1_index, reactant2_index in boundary_birth_indices:
                if (reactant1_index, reactant2_index) in self.process_tracker.boundary_processes:
                    _boundary_process = self.process_tracker.boundary_processes[(reactant1_index, reactant2_index)]
                else:
                    _boundary_process = self.process_tracker.boundary_processes[(reactant2_index, reactant1_index)]
                if _boundary_process.is_used:
                    _reaction_str = f"{self.clusters[reactant1_index]} + {self.clusters[reactant2_index]} -> {_boundary_process.product_label} -> {self.clusters[_boundary_process.boundary_cluster_idx]}"
                    for removed_monomer_idx, removed_mol_count in _boundary_process.removed_monomers.items():
                        _reaction_str += f" + {self.clusters[removed_monomer_idx]} (x{removed_mol_count})"
                    reactions.append(_reaction_str)
        return reactions
    
    def get_boundary_monomer_formation_reactions(self, cluster_idx: int) -> List[str]:
        """Get the boundary monomer formation reactions for a specific cluster."""
        boundary_monomer_birth_indices = self.process_tracker.get_boundary_cluster_monomer_birth_terms(cluster_idx)
        reactions = []
        if boundary_monomer_birth_indices:
            for reactant1_index, reactant2_index, count in boundary_monomer_birth_indices:
                if (reactant1_index, reactant2_index) in self.process_tracker.boundary_processes:
                    _boundary_process = self.process_tracker.boundary_processes[(reactant1_index, reactant2_index)]
                else:
                    _boundary_process = self.process_tracker.boundary_processes[(reactant2_index, reactant1_index)]
                if _boundary_process.is_used:
                    _reaction_str = f"{self.clusters[reactant1_index]} + {self.clusters[reactant2_index]} -> {_boundary_process.product_label} -> {self.clusters[_boundary_process.boundary_cluster_idx]}"                        
                    for removed_monomer_idx, removed_mol_count in _boundary_process.removed_monomers.items():
                        _reaction_str += f" + {self.clusters[removed_monomer_idx]} (x{removed_mol_count})"
                    reactions.append(_reaction_str)
        return reactions
    
    def print_cluster_reactions(self, cluster_idx: int, max_terms: int = 5):
        """Print the cluster tracking information for debugging."""
        cluster = self.clusters[cluster_idx]
        print(f"\nCluster {cluster_idx}: {str(cluster)}")

        # Collision loss terms
        collision_loss_reactions = self.get_collision_loss_reactions(cluster_idx)
        if collision_loss_reactions:
            print("  Collision Loss:")
            self.print_max_reactions_strings(collision_loss_reactions, max_terms, "   ")
        
        outgrowth_loss_reactions = self.get_outgrowth_loss_reactions(cluster_idx)
        if outgrowth_loss_reactions:
            print("  Outgrowth Loss:")
            self.print_max_reactions_strings(outgrowth_loss_reactions, max_terms, "   ")
            
        collision_formation_reactions = self.get_collision_formation_reactions(cluster_idx)
        if collision_formation_reactions:
            print("  Collision Formation:")
            self.print_max_reactions_strings(collision_formation_reactions, max_terms, "   ")

        # Evaporation loss terms
        evaporation_loss_reactions = self.get_evaporation_loss_reactions(cluster_idx)
        if evaporation_loss_reactions:
            print("  Evaporation Loss:")
            self.print_max_reactions_strings(evaporation_loss_reactions, max_terms, "   ")

        # Evaporation formation terms
        evaporation_formation_reactions = self.get_evaporation_formation_reactions(cluster_idx)
        if evaporation_formation_reactions:
            print("  Evaporation Formation:")
            self.print_max_reactions_strings(evaporation_formation_reactions, max_terms, "   ")

        # Boundary collision loss terms
        boundary_death_reactions = self.get_boundary_death_reactions(cluster_idx)
        if boundary_death_reactions:
            print("  Boundary Collision Loss:")
            self.print_max_reactions_strings(boundary_death_reactions, max_terms, "   ")

        # Boundary monomer formation terms
        if self.clusters[cluster_idx].is_monomer:
            boundary_monomer_formation_reactions = self.get_boundary_monomer_formation_reactions(cluster_idx)
            if boundary_monomer_formation_reactions:
                print("  Boundary Monomer Formation:")
                self.print_max_reactions_strings(boundary_monomer_formation_reactions, max_terms, "   ")
        else:
            # Boundary collision formation terms
            boundary_formation_reactions = self.get_boundary_formation_reactions(cluster_idx)
            if boundary_formation_reactions:
                print("  Boundary Collision Formation:")
                self.print_max_reactions_strings(boundary_formation_reactions, max_terms, "   ")

    def print_process_tracking(self, max_clusters: int = 10, max_terms: int = 5):
        """Print the reaction tracking information for debugging.

        Args:
            max_clusters (int, optional): Maximum number of clusters to print. Defaults to 10.
            max_terms (int, optional): Maximum number of terms to print for each reaction. Defaults to 5.
        """
        print("Process Tracking:")
        print("=" * 60)

        for i in range(min(max_clusters, self.n_clusters)):
            self.print_cluster_reactions(i, max_terms=max_terms)

        if self.n_clusters > max_clusters:
            print(f"\n... ({self.n_clusters - max_clusters} more clusters)")

        # Print boundary reactions
        if self.process_tracker.boundary_processes:
            print("\nBoundary Processes:")
            for (i, j), _boundary_process in list(self.process_tracker.boundary_processes.items())[:max_terms]:
                removed_str = " + ".join(
                    f"{self.clusters[idx]} (x{count})"
                    for idx, count in _boundary_process.removed_monomers.items()
                )
                if not _boundary_process.is_used:
                    print(
                        f" Not using boundary collision: {self.clusters[i]} + {self.clusters[j]} -> {_boundary_process.product_label} -> {self.clusters[_boundary_process.boundary_cluster_idx]} + {removed_str}"
                    )
                else:
                    print(
                        f"  {self.clusters[i]} + {self.clusters[j]} -> {_boundary_process.product_label} -> {self.clusters[_boundary_process.boundary_cluster_idx]} + {removed_str}"
                    )
            if len(self.process_tracker.boundary_processes) > max_terms:
                print(
                    f"... ({len(self.process_tracker.boundary_processes) - max_terms} more processes)"
                )
            print()
        # Print outgrowth reactions
        if self.process_tracker.outgrowth_collisions:
            print("\nOutgrowth Processes:")
            for outgrowth_collision in list(self.process_tracker.outgrowth_collisions.values())[:max_terms]:
                i, j = outgrowth_collision.reactant_indices
                comp_str = "".join(
                    [f"{count}{mol}" for mol, count in outgrowth_collision.product_composition.items()]
                )
                print(
                    f"  {self.clusters[i]} + {self.clusters[j]} -> {comp_str} (outgrows)"
                )
            if len(self.process_tracker.outgrowth_collisions) > max_terms:
                print(
                    f"... ({len(self.process_tracker.outgrowth_collisions) - max_terms} more processes)"
                )
            print()

    def get_process_statistics(self) -> Dict[str, int]:
        """Get statistics about the reaction system."""
        stats = self.process_tracker.get_process_statistics()
        return stats

    def _build_formation_rate_and_outflux(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build the formation rate and outflux matrix equations.

        Returns:
            formation_rate: Dict[str, Any]: The dict containing the formation rate equation. Contains the symbolic and callable equations.
            outflux_matrix: Dict[str, Any]: The dict containing the outflux matrix equation. Contains the symbolic and callable equations.
        """
        logger.debug("Building formation rate and outflux matrix equations...")
        start_time = time.time()
        # formation rate is the sum of all collision rates that outgrow and escape the system
        formation_rate_terms = []
        # outflux matrix is final fluxes out of the system in terms of the collisions
        outflux_matrix = sym.zeros(self.n_clusters, self.n_clusters)
        for outgrowth_collision in self.process_tracker.outgrowth_collisions.values():
            i, j = outgrowth_collision.reactant_indices
            collision_coefficient = self.collision_coefficients[i, j]
            if i == j:
                collision_coefficient *= 0.5
            _term = collision_coefficient * self.concentrations[i] * self.concentrations[j]
            outflux_matrix[i, j] += _term
            formation_rate_terms.append(_term)
        
        # add all the terms together (will be 0 if no outgrowth reactions)
        formation_rate: sym.Expr = sym.Add(*formation_rate_terms)

        logger.debug(f"Outflux and Formation rates symbolic equations built in {time.time() - start_time:.3f} seconds")

        outflux_matrix_time = time.time()
        outflux_matrix_function = sym.lambdify(
            (self.concentrations, self.collision_coefficients), outflux_matrix, modules="numpy"
        )
        logger.debug(f"Outflux matrix lambdified in {time.time() - outflux_matrix_time:.3f} seconds")
        
        formation_rate_time = time.time()
        formation_rate_function = sym.lambdify(
            (self.concentrations, self.collision_coefficients), formation_rate, modules="numpy"
        )
        logger.debug(f"Formation rate lambdified in {time.time() - formation_rate_time:.3f} seconds")
        self.formation_rate: SymbolicAndCallableEquations = {
            "symbolic": formation_rate,
            "callable": formation_rate_function,
        }
        self.outflux_matrix: SymbolicAndCallableEquations = {
            "symbolic": outflux_matrix,
            "callable": outflux_matrix_function,
        }
        logger.info(f"Formation rate and outflux matrix equations built in {time.time() - start_time:.3f} seconds")
        return self.formation_rate, self.outflux_matrix
    
    def compute_formation_rate(self, concentrations: np.ndarray, collision_coefficients: np.ndarray) -> np.ndarray:
        if not hasattr(self, "formation_rate"):
            self._build_formation_rate_and_outflux()
        if concentrations.ndim == 2 and concentrations.shape[0] != self.n_clusters:
            concentrations = concentrations.T
        return self.formation_rate["callable"](concentrations, collision_coefficients)

    def compute_outflux_matrix(self, concentrations: np.ndarray, collision_coefficients: np.ndarray) -> np.ndarray:
        if not hasattr(self, "outflux_matrix"):
            self._build_formation_rate_and_outflux()
        return self.outflux_matrix["callable"](concentrations, collision_coefficients)

    def _build_fluxes(self) -> None:
        """Build the net collision-evaporation matrix for the system."""
        logger.debug("Building fluxes equations...")
        fluxes_start = time.time()
        cluster_to_cluster_terms = defaultdict(list)
        cluster_to_neutral_terms = defaultdict(list)
        cluster_to_negative_terms = defaultdict(list)
        cluster_to_positive_terms = defaultdict(list)
        cluster_to_boundary_terms = defaultdict(list)
        boundary_to_cluster_terms = defaultdict(list)
        cluster_to_wall_terms = defaultdict(list)
        cluster_to_coagulation_sink_terms = defaultdict(list)
        source_to_cluster_terms = defaultdict(list)
        cluster_to_dilution_terms = defaultdict(list)

        def _accumulate(accumulator, idx, term):
            """Collect raw terms so we can build symbolic sums in one pass."""
            accumulator[idx].append(term)
        # go through all clusters
        for i in range(self.n_clusters):
            # Collisions removing this cluster i+j -> k
            for j, k in self.process_tracker.get_collision_death_terms(i):
                if k is None:
                    continue
                else:
                    # even if i == j, the rate is 2*0.5 * K[i,j] * C[i] * C[j] = K[i,j] * C[i] * C[j]
                    _accumulate(
                        cluster_to_cluster_terms,
                        (i, k),
                        self.collision_coefficients[i, j]
                        * self.concentrations[i]
                        * self.concentrations[j],
                    )
            # Evaporations resulting in this cluster k -> i+j
            for j, k in self.process_tracker.get_evaporation_birth_terms(i):
                if i == j:  # Need to double evaporation rate for same cluster
                    _accumulate(
                        cluster_to_cluster_terms,
                        (k, i),
                        2 * self.evaporation_coefficients[k, i] * self.concentrations[k],
                    )
                else:
                    _accumulate(
                        cluster_to_cluster_terms,
                        (k, i),
                        self.evaporation_coefficients[k, i] * self.concentrations[k],
                    )

            # Boundary collisions removing this cluster:  i+j -> boundary -> k + (count * monomer)
            for j, k in self.process_tracker.get_boundary_death_terms(i):
                _accumulate(
                    cluster_to_boundary_terms,
                    i,
                    self.collision_coefficients[i, j]
                    * self.concentrations[i]
                    * self.concentrations[j],
                )
            # Boundary collisions forming this cluster:  j + k -> boundary -> i (+ count * monomer)
            for j, k in self.process_tracker.get_boundary_cluster_birth_terms(i):
                if i == j:
                    _accumulate(
                        boundary_to_cluster_terms,
                        i,
                        0.5
                        * self.collision_coefficients[j, k]
                        * self.concentrations[j]
                        * self.concentrations[k],
                    )
                else:
                    _accumulate(
                        boundary_to_cluster_terms,
                        i,
                        self.collision_coefficients[j, k]
                        * self.concentrations[j]
                        * self.concentrations[k],
                    )
            # Boundary monomer formation:  j + k -> boundary -> count * i (monomer_idx) + (cluster_idx)
            for j, k, count in self.process_tracker.get_boundary_cluster_monomer_birth_terms(i):
                if i == j:
                    _accumulate(
                        boundary_to_cluster_terms,
                        i,
                        0.5
                        * count
                        * self.collision_coefficients[j, k]
                        * self.concentrations[j]
                        * self.concentrations[k],
                    )
                else:
                    _accumulate(
                        boundary_to_cluster_terms,
                        i,
                        count
                        * self.collision_coefficients[j, k]
                        * self.concentrations[j]
                        * self.concentrations[k],
                    )
            # External losses
            _accumulate(
                cluster_to_wall_terms,
                i,
                self.wall_losses[i] * self.concentrations[i],
            )
            _accumulate(
                cluster_to_coagulation_sink_terms,
                i,
                self.coagulation_sinks[i] * self.concentrations[i],
            )
            _accumulate(
                cluster_to_dilution_terms,
                i,
                self.dilution_loss * self.concentrations[i],
            )
            # External sources
            _accumulate(source_to_cluster_terms, i, self.sources[i])

        # Add outgrowth fluxes
        for outgrowth_collision in self.process_tracker.outgrowth_collisions.values():
            i, j = outgrowth_collision.reactant_indices
            k_charge = outgrowth_collision.product_charge
            rate = (
                self.collision_coefficients[i, j]
                * self.concentrations[i]
                * self.concentrations[j]
            )
            if k_charge == 0:
                _accumulate(cluster_to_neutral_terms, i, rate)
                _accumulate(cluster_to_neutral_terms, j, rate)
            elif k_charge > 0:
                _accumulate(cluster_to_positive_terms, i, rate)
                _accumulate(cluster_to_positive_terms, j, rate)
            elif k_charge < 0:
                _accumulate(cluster_to_negative_terms, i, rate)
                _accumulate(cluster_to_negative_terms, j, rate)

        def _finalize(shape, accumulator):
            matrix = sym.zeros(*shape)
            for idx, terms in accumulator.items():
                matrix[idx] = (
                    terms[0]
                    if len(terms) == 1
                    else sym.Add(*terms, evaluate=False)
                )
            return matrix

        cluster_to_cluster_flux = _finalize(
            (self.n_clusters, self.n_clusters), cluster_to_cluster_terms
        )
        cluster_to_neutral_flux = _finalize(
            (self.n_clusters, 1), cluster_to_neutral_terms
        )
        cluster_to_negative_flux = _finalize(
            (self.n_clusters, 1), cluster_to_negative_terms
        )
        cluster_to_positive_flux = _finalize(
            (self.n_clusters, 1), cluster_to_positive_terms
        )
        cluster_to_boundary_flux = _finalize(
            (self.n_clusters, 1), cluster_to_boundary_terms
        )
        boundary_to_cluster_flux = _finalize(
            (self.n_clusters, 1), boundary_to_cluster_terms
        )
        cluster_to_wall_flux = _finalize(
            (self.n_clusters, 1), cluster_to_wall_terms
        )
        cluster_to_coagulation_sink_flux = _finalize(
            (self.n_clusters, 1), cluster_to_coagulation_sink_terms
        )
        source_to_cluster_flux = _finalize(
            (self.n_clusters, 1), source_to_cluster_terms
        )
        cluster_to_dilution_flux = _finalize(
            (self.n_clusters, 1), cluster_to_dilution_terms
        )

        # lambdify all fluxes
        all_fluxes = (
            cluster_to_cluster_flux,
            cluster_to_neutral_flux,
            cluster_to_negative_flux,
            cluster_to_positive_flux,
            cluster_to_boundary_flux,
            boundary_to_cluster_flux,
            cluster_to_wall_flux,
            cluster_to_coagulation_sink_flux,
            source_to_cluster_flux,
            cluster_to_dilution_flux,
        )
        fluxes_function = sym.lambdify(
            (
                self.concentrations,
                self.collision_coefficients,
                self.evaporation_coefficients,
                self.wall_losses,
                self.coagulation_sinks,
                self.sources,
                self.dilution_loss,
            ),
            all_fluxes,
            modules="numpy",
        )
        fluxes_time = time.time() - fluxes_start
        logger.debug(f"Fluxes equations built in {fluxes_time:.3f} seconds")
        self.fluxes: SymbolicAndCallableFluxes = {
            "symbolic": OrderedDict(
                {
                    FluxDirection("cluster", "cluster"): cluster_to_cluster_flux,
                    FluxDirection("cluster", "out_neutral"): cluster_to_neutral_flux,
                    FluxDirection("cluster", "out_negative"): cluster_to_negative_flux,
                    FluxDirection("cluster", "out_positive"): cluster_to_positive_flux,
                    FluxDirection("cluster", "boundary"): cluster_to_boundary_flux,
                    FluxDirection("boundary", "cluster"): boundary_to_cluster_flux,
                    FluxDirection("cluster", "wall"): cluster_to_wall_flux,
                    FluxDirection("cluster", "coagulation_sink"): cluster_to_coagulation_sink_flux,
                    FluxDirection("source", "cluster"): source_to_cluster_flux,
                    FluxDirection("cluster", "dilution"): cluster_to_dilution_flux,
                }
            ),
            "callable": fluxes_function,
        }

    def get_net_fluxes_function(self) -> Callable:
        if not hasattr(self, "fluxes"):
            self._build_fluxes()

        # create a wrapper function that calculates the net fluxes
        def net_fluxes_function(
            concentrations: np.ndarray,
            collision_coefficients: np.ndarray,
            evaporation_coefficients: np.ndarray,
            wall_losses: np.ndarray,
            coagulation_sinks: np.ndarray,
            sources: np.ndarray,
            dilution_loss: float,
            calc_final_sources: bool = True,
        ) -> Tuple[OrderedDict[FluxDirection, np.ndarray], np.ndarray]:
            logger.debug("Calculating net fluxes...")
            fluxes_start = time.time()
            final_sources = np.zeros(self.n_clusters)
            # compute all fluxes
            _fluxes: Tuple[np.ndarray, ...] = self.fluxes["callable"](
                concentrations,
                collision_coefficients,
                evaporation_coefficients,
                wall_losses,
                coagulation_sinks,
                sources,
                dilution_loss,
            )
            # map indices to keys
            flux_direction_map = {
                k: i for i, k in enumerate(self.fluxes["symbolic"].keys())
            }
            # calculate net fluxes between clusters
            cluster_to_cluster_fluxes = _fluxes[
                flux_direction_map[FluxDirection("cluster", "cluster")]
            ]
            net_fluxes = np.zeros_like(cluster_to_cluster_fluxes)
            for i in range(self.n_clusters):
                for j in range(i + 1, self.n_clusters):
                    _netflux = (
                        cluster_to_cluster_fluxes[i, j]
                        - cluster_to_cluster_fluxes[j, i]
                    )
                    if _netflux > 0:
                        net_fluxes[i, j] = _netflux
                    else:
                        net_fluxes[j, i] = -_netflux
            boundary_to_cluster_fluxes = _fluxes[
                flux_direction_map[FluxDirection("boundary", "cluster")]
            ].flatten()
            cluster_to_boundary_fluxes = _fluxes[
                flux_direction_map[FluxDirection("cluster", "boundary")]
            ].flatten()
            net_boundary_to_cluster_fluxes = np.zeros_like(boundary_to_cluster_fluxes)
            net_cluster_to_boundary_fluxes = np.zeros_like(cluster_to_boundary_fluxes)
            for i in range(self.n_clusters):
                _netflux = boundary_to_cluster_fluxes[i] - cluster_to_boundary_fluxes[i]
                if _netflux > 0:
                    net_boundary_to_cluster_fluxes[i] = _netflux
                else:
                    net_cluster_to_boundary_fluxes[i] = -_netflux
            # create a dictionary of all net fluxes
            
            # solving source terms of monomers, if needed
            if calc_final_sources:
                monomers = self.clusters.get_monomers()
                for monomer in monomers:
                    monomer_idx = self.clusters.get_index_by_label(monomer.label)
                    from_monomer_flux = (
                        sum(_fluxes[flux_direction_map[FluxDirection("cluster", "cluster")]][monomer_idx,:])
                        + _fluxes[flux_direction_map[FluxDirection("cluster", "out_neutral")]][monomer_idx]
                        + _fluxes[flux_direction_map[FluxDirection("cluster", "out_negative")]][monomer_idx]
                        + _fluxes[flux_direction_map[FluxDirection("cluster", "out_positive")]][monomer_idx]
                        + _fluxes[flux_direction_map[FluxDirection("cluster", "boundary")]][monomer_idx]
                        + _fluxes[flux_direction_map[FluxDirection("cluster", "wall")]][monomer_idx]
                        + _fluxes[flux_direction_map[FluxDirection("cluster", "coagulation_sink")]][monomer_idx]
                        + _fluxes[flux_direction_map[FluxDirection("cluster", "dilution")]][monomer_idx]
                    )
                    to_monomer_flux = (
                        sum(_fluxes[flux_direction_map[FluxDirection("cluster", "cluster")]][:,monomer_idx])
                        + _fluxes[flux_direction_map[FluxDirection("boundary", "cluster")]][monomer_idx]
                        + _fluxes[flux_direction_map[FluxDirection("source", "cluster")]][monomer_idx]
                    )
                    final_sources[monomer_idx] = from_monomer_flux - to_monomer_flux
                    # if no input sources, set the source flux to the final source flux
                    if np.allclose(sources[monomer_idx], 0):
                        _fluxes[flux_direction_map[FluxDirection("source", "cluster")]][monomer_idx] = final_sources[monomer_idx]
                
            all_net_fluxes = OrderedDict(
                {
                    FluxDirection("cluster", "cluster"): net_fluxes, # net fluxes between clusters
                    FluxDirection("cluster", "out_neutral"): _fluxes[
                        flux_direction_map[FluxDirection("cluster", "out_neutral")]
                    ].flatten(),
                    # net fluxes from clusters to neutral outgrowing clusters
                    FluxDirection("cluster", "out_negative"): _fluxes[
                        flux_direction_map[FluxDirection("cluster", "out_negative")]
                    ].flatten(),
                    # net fluxes from clusters to negative outgrowing clusters
                    FluxDirection("cluster", "out_positive"): _fluxes[
                        flux_direction_map[FluxDirection("cluster", "out_positive")]
                    ].flatten(),
                    # net fluxes from clusters to boundary (that lead out of system but are not outgrowing)
                    FluxDirection("cluster", "boundary"): net_cluster_to_boundary_fluxes,
                    # net fluxes from boundary back to clusters in system
                    FluxDirection(
                        "boundary", "cluster"
                    ): net_boundary_to_cluster_fluxes,
                    # net fluxes from clusters to wall loss
                    FluxDirection("cluster", "wall"): _fluxes[
                        flux_direction_map[FluxDirection("cluster", "wall")]
                    ].flatten(),
                    # net fluxes from clusters to coagulation sink
                    FluxDirection("cluster", "coagulation_sink"): _fluxes[
                        flux_direction_map[
                            FluxDirection("cluster", "coagulation_sink")
                        ]
                    ].flatten(),
                    # net fluxes from source to clusters
                    FluxDirection("source", "cluster"): _fluxes[
                        flux_direction_map[FluxDirection("source", "cluster")]
                    ].flatten(),
                    # net fluxes from clusters to dilution
                    FluxDirection("cluster", "dilution"): _fluxes[
                        flux_direction_map[FluxDirection("cluster", "dilution")]
                    ].flatten(),
                }
            )
            net_fluxes_time = time.time() - fluxes_start
            logger.debug(f"Net fluxes calculated in {net_fluxes_time:.3f} seconds")
            return all_net_fluxes, final_sources

        return net_fluxes_function

    def _format_perl_style_equation(self, symbolic_equation):
        import re

        eq_str = str(symbolic_equation)
        eq_str = re.sub(
            r"K\\[(\\d+), (\\d+)\\]",
            lambda m: f"K({int(m.group(1)) + 1},{int(m.group(2)) + 1})",
            eq_str,
        )
        eq_str = re.sub(r"C_(\\d+)", lambda m: f"c({int(m.group(1)) + 1})", eq_str)
        eq_str = eq_str.replace(" + ", " +\n\t")
        return eq_str

    def satisfies_outgrowth_criteria(
        self, composition: Dict[str, int], charge: int
    ) -> bool:
        """Check if cluster satisfies outgrowth criteria to leave system."""

        # Check charge-specific outgrowth rules
        rules_to_check = []
        if charge in self.equations_config.outgrowth_rules_by_charge:
            rules_to_check.extend(self.equations_config.outgrowth_rules_by_charge[charge])
        else:
            logger.warning(f"No outgrowth rules for charge {charge}. Assuming cluster will not outgrow.")
            return False

        # Check specific outgrowth rules
        for rule in rules_to_check:
            if self._matches_rule(composition, charge, rule):
                return True

        return False

    def _matches_rule(
        self, composition: Dict[str, int], charge: int, rule: Dict[str, int]
    ) -> bool:
        """Check if composition matches a specific outgrowth rule."""
        for mol, min_count in rule.items():
            if composition.get(mol, 0) < min_count:
                return False
        return True

    def bring_to_boundary(
        self, composition: Dict[str, int], charge: int
    ) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
        """Try to bring cluster back to system boundary by removing molecules.

        This is a full translation of the Perl check_boundary subroutine logic.
        """
        removed_mols = {}
        boundary_comp = composition.copy()

        # Step 1: Remove excess molecules (equivalent to Perl Step 4a)
        boundary_comp, removed_mols = self._remove_excess_molecules(
            boundary_comp, charge, removed_mols
        )

        # Check if now valid
        if self.clusters.check_if_in_collection(boundary_comp):
            return boundary_comp, removed_mols

        # Step 2: Use acid/base strength logic if available (equivalent to Perl Step 4b)
        if self.equations_config.use_acid_base_logic:
            boundary_comp, removed_mols = self._apply_acid_base_logic(
                boundary_comp, charge, removed_mols
            )

            # Check if now valid
            if self.clusters.check_if_in_collection(boundary_comp):
                return boundary_comp, removed_mols

        # Step 3: Simple molecule removal (equivalent to Perl Step 4c)
        boundary_comp, removed_mols = self._simple_molecule_removal(
            boundary_comp, charge, removed_mols
        )

        # Final check
        if self.clusters.check_if_in_collection(boundary_comp):
            return boundary_comp, removed_mols

        # If we still can't bring it back, return None
        return None, None

    def _remove_excess_molecules(
        self, composition: Dict[str, int], charge: int, removed_mols: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Remove molecules that exceed maximum allowed for the cluster's charge type."""
        boundary_comp = composition.copy()

        # Get max limits for this charge type
        max_limits = {}
        if charge == 0:
            max_limits = self.max_molecules_by_charge.get(0, {})
        elif charge < 0:
            max_limits = self.max_molecules_by_charge.get(-1, {})
        else:  # charge > 0
            max_limits = self.max_molecules_by_charge.get(1, {})

        # Remove excess molecules
        for mol, count in boundary_comp.items():
            if mol in max_limits and count > max_limits[mol]:
                excess = count - max_limits[mol]
                boundary_comp[mol] = max_limits[mol]
                removed_mols[mol] = removed_mols.get(mol, 0) + excess

        return boundary_comp, removed_mols

    def _apply_acid_base_logic(
        self, composition: Dict[str, int], charge: int, removed_mols: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Apply acid/base strength logic for molecule removal."""
        boundary_comp = composition.copy()
        molecules = self.molecules

        # Count acids, bases, and neutrals
        n_acid = 0  # Acids that can be lost
        n_base = 0  # Bases that can be lost
        n_acid_other = 0  # Acids that can't be lost
        n_base_other = 0  # Bases that can't be lost
        n_neutral = 0  # Neutral molecules

        for mol, count in boundary_comp.items():
            if count > 0:
                mol_obj = molecules.get_by_symbol(mol)
                mol_charge = mol_obj.charge
                if mol_charge == 0:
                    n_neutral += count
                acid_strength = getattr(mol_obj, "acid_strength", 0)
                base_strength = getattr(mol_obj, "base_strength", 0)
                can_be_lost = getattr(mol_obj, "can_be_lost_boundary", True)
                if acid_strength > 0:
                    if can_be_lost or mol == "P":  # P is proton
                        n_acid += count
                    else:
                        n_acid_other += count
                if base_strength > 0:
                    if can_be_lost or mol == "MP":  # MP is missing proton
                        n_base += count
                    else:
                        n_base_other += count

        # Decision logic: remove acids or bases first?
        if n_acid + n_acid_other > n_base + n_base_other:
            remove_acids_first = True
        else:
            remove_acids_first = False

        # Removal loop
        acid_min = 0
        base_min = 0
        diff = 0

        while not self.clusters.check_if_in_collection(boundary_comp) and n_neutral > diff:
            if remove_acids_first:
                # Remove weakest acid first
                mol_to_remove = self._find_weakest_acid(boundary_comp, acid_min)
                if mol_to_remove is not None:
                    boundary_comp[mol_to_remove] -= 1
                    if boundary_comp[mol_to_remove] == 0:
                        del boundary_comp[mol_to_remove]

                    removed_mols[mol_to_remove] = removed_mols.get(mol_to_remove, 0) + 1
                    n_neutral -= 1

                    # Update acid count
                    mol_obj = molecules.get_by_symbol(mol_to_remove)
                    if getattr(mol_obj, "acid_strength", 0) > 0:
                        n_acid -= 1
                    if getattr(mol_obj, "base_strength", 0) > 0:
                        n_base -= 1

                    # Check if we should switch to removing bases
                    if n_acid + n_acid_other <= n_base + n_base_other + diff:
                        remove_acids_first = False
                        base_min = 0
                else:
                    acid_min += 1
                    if acid_min > self._get_max_acid_strength():
                        break
            else:
                # Remove weakest base first
                mol_to_remove = self._find_weakest_base(boundary_comp, base_min)
                if mol_to_remove is not None:
                    boundary_comp[mol_to_remove] -= 1
                    if boundary_comp[mol_to_remove] == 0:
                        del boundary_comp[mol_to_remove]

                    removed_mols[mol_to_remove] = removed_mols.get(mol_to_remove, 0) + 1
                    n_neutral -= 1

                    # Update base count
                    mol_obj = molecules.get_by_symbol(mol_to_remove)
                    if getattr(mol_obj, "base_strength", 0) > 0:
                        n_base -= 1
                    if getattr(mol_obj, "acid_strength", 0) > 0:
                        n_acid -= 1

                    # Check if we should switch to removing acids
                    if n_acid + n_acid_other + diff > n_base + n_base_other:
                        remove_acids_first = True
                        acid_min = 0
                else:
                    base_min += 1
                    if base_min > self._get_max_base_strength():
                        break

        return boundary_comp, removed_mols

    def _find_weakest_acid(
        self, composition: Dict[str, int], min_strength: int
    ) -> Optional[str]:
        """Find the molecule with the lowest acid strength >= min_strength."""
        candidates = []
        molecules = self.molecules
        for mol, count in composition.items():
            if count > 0:
                mol_obj = molecules.get_by_symbol(mol)
                acid_strength = getattr(mol_obj, "acid_strength", 0)
                charge = mol_obj.charge
                can_be_lost = getattr(mol_obj, "can_be_lost_boundary", True)

                if (
                    acid_strength >= min_strength
                    and charge == 0
                    and (can_be_lost or mol == "P")
                ):
                    candidates.append((mol, acid_strength))

        if candidates:
            # Return the one with lowest acid strength
            return min(candidates, key=lambda x: x[1])[0]
        return None

    def _find_weakest_base(
        self, composition: Dict[str, int], min_strength: int
    ) -> Optional[str]:
        """Find the molecule with the lowest base strength >= min_strength."""
        candidates = []
        molecules = self.molecules
        for mol, count in composition.items():
            if count > 0:
                mol_obj = molecules.get_by_symbol(mol)
                base_strength = getattr(mol_obj, "base_strength", 0)
                charge = mol_obj.charge
                can_be_lost = getattr(mol_obj, "can_be_lost_boundary", True)

                if (
                    base_strength >= min_strength
                    and charge == 0
                    and (can_be_lost or mol == "MP")
                ):
                    candidates.append((mol, base_strength))

        if candidates:
            # Return the one with lowest base strength
            return min(candidates, key=lambda x: x[1])[0]
        return None

    def _get_max_acid_strength(self) -> int:
        """Get the maximum acid strength."""
        max_strength = 0
        molecules = self.molecules
        for mol_obj in molecules:
            max_strength = max(max_strength, getattr(mol_obj, "acid_strength", 0))
        return max_strength

    def _get_max_base_strength(self) -> int:
        """Get the maximum base strength."""
        max_strength = 0
        molecules = self.molecules
        for mol_obj in molecules:
            max_strength = max(max_strength, getattr(mol_obj, "base_strength", 0))
        return max_strength

    def _simple_molecule_removal(
        self, composition: Dict[str, int], charge: int, removed_mols: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Simple molecule removal as fallback (equivalent to Perl Step 4c)."""
        boundary_comp = composition.copy()
        molecules = self.molecules

        while not self.clusters.check_if_in_collection(boundary_comp):
            # Find the most numerous removable neutral molecule
            mol_to_remove = None
            max_count = 0

            for mol, count in boundary_comp.items():
                if count > 0:
                    mol_obj = molecules.get_by_symbol(mol)
                    charge_mol = mol_obj.charge
                    can_be_lost = getattr(mol_obj, "can_be_lost_boundary", True)

                    if can_be_lost and charge_mol == 0:
                        # Special handling for charged clusters with protons/missing protons
                        removable_count = count

                        # If cluster is negative and has missing proton, adjust counts
                        if charge < 0 and "MP" in boundary_comp and mol != "MP":
                            mp_count = boundary_comp["MP"]
                            removable_count = max(0, count - mp_count)

                        # If cluster is positive and has proton, adjust counts
                        elif charge > 0 and "P" in boundary_comp and mol != "P":
                            p_count = boundary_comp["P"]
                            removable_count = max(0, count - p_count)

                        if removable_count > max_count:
                            max_count = removable_count
                            mol_to_remove = mol

            if mol_to_remove is None:
                break  # Can't remove any more molecules

            # Remove one molecule
            boundary_comp[mol_to_remove] -= 1
            if boundary_comp[mol_to_remove] == 0:
                del boundary_comp[mol_to_remove]

            removed_mols[mol_to_remove] = removed_mols.get(mol_to_remove, 0) + 1

        return boundary_comp, removed_mols

    def update_equations_config(self, **kwargs):
        """Update the equations configuration."""
        flag_rebuild = False
        for key, value in kwargs.items():
            if hasattr(self.equations_config, key):
                logger.info(f"Updating EquationsConfiguration parameter: {key} to {value}")
                setattr(self.equations_config, key, value)
                flag_rebuild = True
            else:
                raise ValueError(f"Unknown EquationsConfiguration parameter: {key}")
        
        if flag_rebuild:
            logger.info("Rebuilding equations...")
            self.rebuild()
    
    def to_config(self) -> Dict[str, Any]:
        """Get the equations configuration."""
        return self.equations_config.to_config()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], clusters: ClusterCollection) -> "ClusterEquations":
        """Create an equations configuration from a configuration."""
        equations_config = EquationsConfiguration.from_config(config)
        return cls(clusters, equations_config)
    
    def _map_idx_column_to_label(self, df: pd.DataFrame, column_name: str) -> pd.Series:
        """Map a column to a label."""
        return df[column_name].apply(lambda x: self.clusters[x].label)
    
    def _map_dataframe_idx_to_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map a dataframe to a label."""
        idx_columns = [col for col in df.columns if col.endswith("_idx")]
        for col in idx_columns:
            new_col = col.replace("_idx", "_label")
            df[new_col] = self._map_idx_column_to_label(df, col)
        return df
    
    def get_cluster_collision_loss_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the collision loss dataframe for a cluster."""
        df =  self.process_tracker.get_collision_loss_dataframe(cluster_idx)
        df = self._map_dataframe_idx_to_label(df)
        return df
    
    def get_cluster_evaporation_loss_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the evaporation loss dataframe for a cluster."""
        df = self.process_tracker.get_evaporation_loss_dataframe(cluster_idx)
        df = self._map_dataframe_idx_to_label(df)
        return df
    
    def get_cluster_collision_formation_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the collision formation dataframe for a cluster."""
        df = self.process_tracker.get_collision_formation_dataframe(cluster_idx)
        df = self._map_dataframe_idx_to_label(df)
        return df
    
    def get_cluster_evaporation_formation_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the evaporation formation dataframe for a cluster."""
        df = self.process_tracker.get_evaporation_formation_dataframe(cluster_idx)
        df = self._map_dataframe_idx_to_label(df)
        return df
    
    def get_cluster_boundary_loss_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the boundary death dataframe for a cluster."""
        df = self.process_tracker.get_boundary_death_dataframe(cluster_idx)
        df = self._map_dataframe_idx_to_label(df)
        return df
    
    def get_cluster_boundary_cluster_formation_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the boundary cluster formation dataframe for a cluster."""
        df = self.process_tracker.get_boundary_cluster_formation_dataframe(cluster_idx)
        df = self._map_dataframe_idx_to_label(df)
        return df
    
    def get_cluster_boundary_monomer_formation_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the boundary monomer formation dataframe for a cluster."""
        df = self.process_tracker.get_boundary_monomer_formation_dataframe(cluster_idx)
        df = self._map_dataframe_idx_to_label(df)
        return df
    
    def get_cluster_outgrowth_loss_dataframe(self, cluster_idx: int) -> pd.DataFrame:
        """Get the outgrowth loss dataframe for a cluster."""
        df = self.process_tracker.get_outgrowth_death_dataframe(cluster_idx)
        df = self._map_dataframe_idx_to_label(df)
        if "product_composition" in df.columns:
            df["product_label"] = df["product_composition"].apply(lambda x: self.clusters.composition_to_label(x))
        return df