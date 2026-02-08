"""
Molecular species definitions for ACDC.
"""

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ..utils.constants import parse_quantity, ureg


class MoleculeType(Enum):
    """Types of molecular species."""

    NEUTRAL = "neutral"
    POSITIVE_ION = "positive_ion"
    NEGATIVE_ION = "negative_ion"
    PROTON = "proton"
    MISSING_PROTON = "missing_proton"


@dataclass
class Molecule:
    """Represents a molecular species in the ACDC system."""

    name: str
    symbol: str
    charge: int
    mass: float  # kg
    density: Optional[float] = None  # kg/m³
    base_strength: int = 0
    acid_strength: int = 0

    # Optional fields for ions
    corresponding_neutral: Optional[str] = None
    corresponding_negative_ion: Optional[str] = None
    corresponding_positive_ion: Optional[str] = None
    can_be_lost_boundary: bool = True

    def __post_init__(self):
        """Validate molecule data after initialization."""
        if self.mass <= 0:
            raise ValueError(f"Mass must be positive, got {self.mass}")
        if self.density is not None and self.density <= 0:
            raise ValueError(f"Density must be positive, got {self.density}")

        # Determine molecule type based on charge
        if self.charge == 0:
            self.type = MoleculeType.NEUTRAL
        elif self.charge == 1:
            if self.symbol == "P" or self.name == "proton":
                self.type = MoleculeType.PROTON
                self.density = 0
            else:
                self.type = MoleculeType.POSITIVE_ION
        elif self.charge == -1:
            if self.symbol == "MP" or self.name == "missing_proton":
                self.type = MoleculeType.MISSING_PROTON
                self.density = 0
            else:
                self.type = MoleculeType.NEGATIVE_ION
        else:
            raise ValueError(f"Invalid charge {self.charge} for molecule {self.name}")

    @property
    def volume(self) -> float:
        """Molecular volume in m³."""
        return self.mass / self.density if self.density is not None else 0

    def __str__(self) -> str:
        return f"{self.name} ({self.symbol}, charge={self.charge})"

    def __repr__(self) -> str:
        return f"Molecule(name='{self.name}', symbol='{self.symbol}', charge={self.charge})"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Molecule":
        """Create a molecule from a dictionary."""
        return cls(
            name=data["name"],
            symbol=data["symbol"],
            charge=data["charge"],
            # Convert mass to SI units
            mass=parse_quantity(
                data["mass"], default_unit="g/mol", target_unit="kg/particle"
            ),
            # Convert density to SI units
            density=parse_quantity(data["density"], default_unit="kg/m^3") if "density" in data else None,
            # Needed for outgrowth rules
            base_strength=data.get("base_strength", 0),
            acid_strength=data.get("acid_strength", 0),
            corresponding_neutral=data.get("corresponding_neutral"),
            corresponding_negative_ion=data.get("corresponding_negative_ion"),
            corresponding_positive_ion=data.get("corresponding_positive_ion"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the molecule to a dictionary."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "charge": self.charge,
            "mass": f"{(self.mass * ureg('kg/particle').to('g/mol')).magnitude} g/mol",
            "density": f"{(self.density * ureg('kg/m^3')).magnitude} kg/m^3" if self.density is not None else None,
            "base_strength": self.base_strength,
            "acid_strength": self.acid_strength,
            "corresponding_neutral": self.corresponding_neutral,
            "corresponding_negative_ion": self.corresponding_negative_ion,
            "corresponding_positive_ion": self.corresponding_positive_ion,
        }


class MoleculeCollection:
    """Collection of molecules with lookup capabilities."""

    def __init__(self, molecules: OrderedDict[str, Molecule]):
        self.molecules = molecules
        self._by_symbol = {mol.symbol: mol for mol in molecules.values()}
        self._by_name = {mol.name: mol for mol in molecules.values()}

        # Store molecule order for consistent composition handling
        self._molecule_order = list(molecules.keys())

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MoleculeCollection":
        """Create a molecule collection from a configuration dictionary."""
        molecules = OrderedDict()
        for symbol, mol_data in config.items():
            mol_data["symbol"] = symbol
            molecule = Molecule.from_dict(mol_data)
            molecules[symbol] = molecule
        return cls(molecules)

    def to_config(self) -> Dict[str, Any]:
        """Convert the molecule collection to a YAML-style config."""
        config = {}
        config = {symbol: mol.to_dict() for symbol, mol in self.molecules.items()}
        return config

    def get_molecule_order(self) -> List[str]:
        """Get the order of molecules as defined in the collection."""
        return self._molecule_order.copy()

    def create_empty_composition(self) -> OrderedDict[str, int]:
        """Create an empty composition dict with all molecules set to 0."""
        return OrderedDict((symbol, 0) for symbol in self._molecule_order)

    def get_by_symbol(self, symbol: str) -> Molecule:
        """Get molecule by symbol."""
        if symbol not in self._by_symbol:
            raise KeyError(f"Molecule with symbol '{symbol}' not found")
        return self._by_symbol[symbol]

    def get_by_name(self, name: str) -> Molecule:
        """Get molecule by name."""
        if name not in self._by_name:
            raise KeyError(f"Molecule with name '{name}' not found")
        return self._by_name[name]

    def get_neutral_molecules(self) -> Dict[str, Molecule]:
        """Get all neutral molecules."""
        return {
            k: v for k, v in self.molecules.items() if v.type == MoleculeType.NEUTRAL
        }

    def get_positive_ions(self) -> Dict[str, Molecule]:
        """Get all positive ions."""
        return {
            k: v
            for k, v in self.molecules.items()
            if v.type == MoleculeType.POSITIVE_ION
        }

    def get_negative_ions(self) -> Dict[str, Molecule]:
        """Get all negative ions."""
        return {
            k: v
            for k, v in self.molecules.items()
            if v.type == MoleculeType.NEGATIVE_ION
        }
   
    def get_corresponding_negative_ion(self, molecule: str|Molecule) -> str|None:
        """Get the corresponding negative ion for a molecule."""
        if isinstance(molecule, str):
            molecule = self.molecules[molecule]
        return molecule.corresponding_negative_ion
    
    def get_corresponding_positive_ion(self, molecule: str|Molecule) -> str|None:
        """Get the corresponding positive ion for a molecule."""
        if isinstance(molecule, str):
            molecule = self.molecules[molecule]
        return molecule.corresponding_positive_ion
    
    def get_corresponding_neutral(self, molecule: str|Molecule) -> str|None:
        """Get the corresponding neutral molecule for a molecule."""
        if isinstance(molecule, str):
            molecule = self.molecules[molecule]
        return molecule.corresponding_neutral
    
    def __len__(self) -> int:
        return len(self.molecules)

    def __iter__(self):
        return iter(self.molecules.values())

    def __getitem__(self, key: str) -> Molecule:
        return self.molecules[key]

    def __contains__(self, key: str) -> bool:
        return key in self.molecules