from pathlib import Path

from eduacdc.io import InputParser
from eduacdc.simulation import Simulation


# Settings
config_path = Path(__file__).resolve().parent / "ADW.yaml"

temperature = 225.0
humidity = 0.5

initial_conditions = {
    "1A": {"type": "constant", "value": "1e7 cm^-3"},
    "1D": {"type": "constant", "value": "100 ppt"},}


# Read the system
parser = InputParser()
system = parser.get_system_from_yaml(config_path)

# Set enhancement factors to 2
system.set_enhancements(2)


# Create the equations
equations = parser.get_equations_from_yaml(config_path, system.clusters)

# Create the simulation
simulation = Simulation(system, equations)


# Run the steady-state simulation
results = simulation.run_steady_state(
    initial_conditions=initial_conditions,
    temperature=temperature,
    relative_humidity=humidity,
    max_time=1e8,
    tolerance=1e-6,)


# Get the formation rate
J = results.get_final_formation_rate("cm^-3 s^-1")

print(f"Steady-state formation rate: {J:.3e} cm^-3 s^-1")
