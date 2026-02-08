# %%
"""
Physical constants used in ACDC calculations.
"""

import pint
from scipy.constants import Avogadro, Boltzmann, gas_constant, pi

# Create a centralized unit registry
ureg = pint.UnitRegistry()
ureg.setup_matplotlib(True)
# define custom units
ureg.define("ppt = 1e-12")


conc = pint.Context(
    name="concentration",
    aliases=["conc"],
    defaults={
        "temperature": 298.15 * ureg.kelvin,
        "pressure": 1 * ureg.atm,
    },
)
# convert from number density to concentration
conc.add_transformation(
    "1/[volume]",
    "[concentration]",
    lambda ureg, x, temperature, pressure: x * ureg("1 particle"),
)
# convert from number density to pressure
conc.add_transformation(
    "1/[volume]",
    "[pressure]",
    lambda ureg, x, temperature, pressure: x * ureg.boltzmann_constant * temperature,
)
# convert from ratio (ppt, ppb, ppm) to number density
conc.add_transformation(
    "[]",
    "1/[volume]",
    lambda ureg, x, temperature, pressure: x
    * pressure
    / (temperature * ureg.boltzmann_constant),
)

conc.add_transformation(
    "1/[volume]",
    "[]",
    lambda ureg, x, temperature, pressure: x
    * (temperature * ureg.boltzmann_constant)
    / pressure
)

ureg.add_context(conc)
# %%
# Add custom units for atmospheric chemistry

# %%
# Fundamental constants
AVOGADRO_CONSTANT = Avogadro  # mol^-1
BOLTZMANN_CONSTANT = Boltzmann  # J/K
GAS_CONSTANT = gas_constant  # J/(mol·K)
PI = pi

# Unit conversions
CAL_TO_J = 4.184
KCAL_TO_J = CAL_TO_J * 1000
CAL_PER_MOL_TO_J = CAL_TO_J / Avogadro
KCAL_PER_MOL_TO_J = KCAL_TO_J / Avogadro  # J
G_TO_KG = 1e-3  # kg/g
G_PER_MOL_TO_KG = 1e-3 / Avogadro  # kg
CM3_TO_M3 = 1e-6  # m³/cm³
PA_TO_ATM = 1.01325e5  # Pa/atm
J_TO_KCAL_PER_MOL = Avogadro / KCAL_TO_J  # J -> kcal/mol
ATMOSPHERIC_PRESSURE = 101325  # Pa
PPT_TO_MOL_M3 = 1e-12  # ppt -> mol/m³
CM3_TO_M3 = 1e-6  # m³/cm³
PER_CM3_TO_PER_M3 = 1e6  # cm^-3 -> m^-3

# Air properties (at 298.15 K, 1 atm)
AIR_VISCOSITY = 1.81e-5  # Pa·s
AIR_DENSITY = 1.225  # kg/m³
AIR_MEAN_FREE_PATH = 6.5e-8  # m

# Water properties
WATER_MASS = 18.01528  # g/mol
WATER_DENSITY = 998.0  # kg/m³
WATER_SURFACE_TENSION = 0.0728  # N/m (at 298.15 K)

# Collision theory parameters
STICKING_COEFFICIENT = 1.0  # Default sticking coefficient
DIFFUSION_ENHANCEMENT = 1.0  # Default diffusion enhancement factor


# Unit parsing and conversion utilities
def parse_quantity(value, default_unit=None, target_unit=None) -> float:
    """
    Parse a quantity that can be a number (assumed SI) or string with units.

    Parameters
    ----------
    value : Union[float, int, str]
        The value to parse. If string, should include units (e.g., "278 K").
        If number, assumed to be in SI units.
    default_unit : str, optional
        Default unit to assume if value is a number. If None, no conversion.
    target_unit : str, optional
        Target unit to convert to. If None, converts to SI base units.

    Returns
    -------
    float
        The value in the target unit (or SI base units if target_unit is None).

    Examples
    --------
    >>> parse_quantity("278 K")
    278.0
    >>> parse_quantity(278, "K")
    278.0
    >>> parse_quantity("1e6 cm^-3", target_unit="1/m^3")
    1000000000.0
    """
    if isinstance(value, str):
        # Parse string with units
        quantity = ureg(value)
        if target_unit:
            return quantity.to(target_unit).magnitude
        else:
            return quantity.to_base_units().magnitude
    elif isinstance(value, (int, float)):
        # Number - assume SI or apply default unit
        if default_unit:
            quantity = value * ureg(default_unit)
            if target_unit:
                return quantity.to(target_unit).magnitude
            else:
                return quantity.to_base_units().magnitude
        else:
            return float(value)
    else:
        raise ValueError(f"Cannot parse quantity: {value}")


def format_quantity(value, unit, precision=3) -> str:
    """
    Format a quantity with units for display.

    Parameters
    ----------
    value : float
        The value in SI units
    unit : str
        The unit to display
    precision : int, optional
        Number of decimal places

    Returns
    -------
    str
        Formatted string with value and unit
    """
    quantity = value * ureg(unit)
    return f"{quantity:~P.{precision}f}"


