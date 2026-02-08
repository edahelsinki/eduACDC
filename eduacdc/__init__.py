"""
eduACDC - Educational Atmospheric Cluster Dynamics Code

An educational Python implementation of the ACDC outgrowth simulation tool.
"""

from .core.clusters import Cluster, ClusterType
from .core.equations import ClusterEquations, EquationsConfiguration
from .core.molecules import Molecule, MoleculeType
from .core.process_coefficients import ProcessCoefficients
from .core.system import SimulationSystem, AmbientConditions
from .io.parser import InputParser
from .simulation.results import SimulationResults
from .simulation.solver import Simulation

__version__ = "0.1.0"
__author__ = "ACDC Python Team"

__all__ = [
    "Molecule",
    "MoleculeType", 
    "Cluster",
    "ClusterType",
    "ProcessCoefficients",
    "ClusterEquations",
    "EquationsConfiguration",
    "SimulationSystem",
    "AmbientConditions",
    "Simulation",
    "InputParser",
    "SimulationResults",
] 