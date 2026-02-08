"""Run a simple eduACDC outgrowth simulation."""
import logging
from pathlib import Path

import yaml

from eduacdc.core import CoagulationLossCalculator, WallLossCalculator
from eduacdc.io import InputParser
from eduacdc.simulation import Simulation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]:%(name)s:%(message)s")

# Set DEBUG level only for eduacdc modules
logging.getLogger("eduacdc").setLevel(logging.INFO)

# Paths relative to script location (run from project root: python examples/run_simulation.py)
examples_dir = Path(__file__).resolve().parent
config_path = examples_dir / "AN_neutral_neg_pos.yaml"

# Load system and equations using InputParser
parser = InputParser()
system = parser.get_system_from_yaml(config_path)
## Check SimulationSystem
system.print_summary()
# Check Equations
equations = parser.get_equations_from_yaml(config_path, system.clusters)
equations.print_process_tracking()

with open(config_path) as f:
    data = yaml.safe_load(f)
initial_conditions = data.get("simulation", {}).get("initial_conditions", {
    "1A": {"type": "constant", "value": "1e6 cm^-3"},
    "1N": {"type": "constant", "value": "1 ppt"},
})

# Run simulation
simulation = Simulation(system, equations)
# Create wall loss calculators
wall_loss_calculator = WallLossCalculator.create_cloud4_ja()
# Create coagulation loss calculator
coagulation_loss = CoagulationLossCalculator.create_exponential()
# Run simulation
results = simulation.run_steady_state(
    initial_conditions,
    wall_loss_calculator=wall_loss_calculator,
    coagulation_loss_calculator=coagulation_loss,
)

# Output
labels = results.system.clusters.get_labels()
concentrations = results.get_final_concentrations(output_units="cm^-3")
print("Final concentrations (cm^-3):")
for label, conc in zip(labels, concentrations):
    print(f"  {label}: {conc:.4e}")
