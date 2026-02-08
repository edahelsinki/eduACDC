"""
Input file parser for ACDC.
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from ..core.clusters import ClusterCollection
from ..core.equations import ClusterEquations
from ..core.system import SimulationSystem


class InputParser:
    """Parser for ACDC input files."""

    def __init__(self):
        pass

    def _parse_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse YAML format input file."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return data

    def get_system_from_yaml(self, file_path: Union[str, Path]) -> SimulationSystem:
        """Parse a system YAML file and return system data."""
        file_path = Path(file_path)
        data = self._parse_yaml_file(file_path)
        return SimulationSystem.from_config(data)

    def get_equations_from_yaml(
        self, file_path: Union[str, Path], clusters: ClusterCollection
        ) -> ClusterEquations:
        """Parse a simulation YAML file and return simulation data."""
        file_path = Path(file_path)
        data = self._parse_yaml_file(file_path)
        equations_data = data.get("equations_configuration", {})
        return ClusterEquations.from_config(equations_data, clusters)