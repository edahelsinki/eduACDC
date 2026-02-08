"""
Wall loss coefficient calculations for a system.
Supports multiple parameterizations similar to the Perl ACDC script.
"""

import logging
import math
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from ..utils.constants import BOLTZMANN_CONSTANT, ureg
from .cluster_properties import ClusterProperties
from .clusters import ClusterCollection
from .system import AmbientConditions

logger = logging.getLogger(__name__)

DEFAULT_FACTOR_FOR_WALL_LOSS_ION_ENHANCEMENT = 3.3 # From ACDC Perl/Matlab standard

class WallLossType(Enum):
    """Types of wall loss parameterizations."""

    DIFFUSION = "diffusion"
    CLOUD4_JA = "cloud4_JA"
    CLOUD4_JK = "cloud4_JK"
    CLOUD4_AK = "cloud4_AK"
    CLOUD4_SIMPLE = "cloud4_simple"
    CLOUD3 = "cloud3"
    IFT = "ift"
    CONSTANT = "constant"
    EXTERNAL = "external"


class WallLossModel(ABC):
    """Abstract base class for wall loss calculation models."""
    @property
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Get the name of the wall loss model."""
        pass
    
    @abstractmethod
    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate wall loss coefficients for all clusters."""
        pass

class DiffusionWallLoss(WallLossModel):
    """Diffusion wall loss for a flow tube (originally for the tube of Hanson and Eisele, 2000)
    Diffusion coefficients are calculated as in the kinetic gas theory, and relative to the coefficient of sulfuric acid
    """

    @property
    def name(self) -> str:
        return self._name

    def __init__(
        self,
        flow_tube_radius: float = 0.049 / 2,
        flow_tube_pressure: float = ureg("620 torr").to("Pa").magnitude,
        temperature: float = 298.15,
    ):
        """
        Initialize diffusion wall loss model.

        Args:
            tube_radius: Tube radius in meters. If None, the default radius of 0.049/2 m is used.
            temperature: Temperature in Kelvin
            pressure: Pressure in Pascal. Default is 620 torr, which is the pressure used in the Hanson and Eisele, 2000 paper.
        """
        self._name = f"{WallLossType.DIFFUSION.value}_using_tube_radius_{flow_tube_radius}m_and_pressure_{flow_tube_pressure}Pa_and_temperature_{temperature}K"
        self.flow_tube_radius = flow_tube_radius
        self.flow_tube_pressure = flow_tube_pressure
        self.temperature = temperature
        self.D_to_wl = (
            3.65 / (self.flow_tube_radius**2)
        )  # for laminar diffusion-limited flow (Brown, R.L.: Tubular flow reactor with first-order kinetics, J. Res. Natl. Bur. Stand. (U.S.), 83, 1, 1-6, 1978)
        # Properties of N2, the radius is calculated from viscosity (e.g. eq. 11-67 in Present: Kinetic theory of gases, 1958)
        self.mass_N2 = (
            28.01 * ureg("g/mol").to("kg/particle").magnitude
        )  # in kg/particle
        self.compute_temperature_dependent_properties()

    def compute_temperature_dependent_properties(self):
        """Compute the temperature-dependent properties of N2."""
        # Sutherland formula, coefficients for N2:
        # C=111 from Crane (1988) (haven't seen the original paper, but lots of references citing the number)
        # mu0=17.9 at T0=300 from CRC
        self.viscosity_N2 = (
            17.9
            * 1e-6
            * (300 + 111)
            / (self.temperature + 111)
            * (self.temperature / 300) ** 1.5
        )
        self.radius_N2 = (
            0.5
            * (5 / 16 / self.viscosity_N2) ** 0.5
            * (self.mass_N2 * BOLTZMANN_CONSTANT * self.temperature / math.pi) ** 0.25
        )

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate wall loss coefficients using diffusion theory."""
        self.temperature = conditions.temperature
        self.compute_temperature_dependent_properties()
        n_clusters = len(clusters)
        # find the acid monomer
        mass1 = None
        radius = None
        for molecule in clusters.molecules:
            if (
                molecule.symbol == "H2SO4"
                or molecule.symbol == "SA"
                or molecule.symbol == "A"
            ) and clusters.check_if_in_collection({molecule.symbol: 1}):
                mass1 = molecule.mass
                radius = molecule.radius
                break
            elif (
                re.search(r"acid", molecule.name, re.IGNORECASE)
                and re.search(r"sulphuric", molecule.name, re.IGNORECASE)
                and clusters.check_if_in_collection({molecule.symbol: 1})
            ):
                logger.info(
                    f"Using {molecule.name} as the acid monomer for diffusion coefficients"
                )
                mass1 = molecule.mass
                radius = molecule.radius
                break

        if mass1 is None or radius is None:
            raise ValueError(
                "Sulfuric acid monomer not found in the cluster collection. Can't calculate diffusion coefficients."
            )

        # Loss coefficient of the acid monomer
        # Diffusion coefficient of acid in N2 (e.g. eq. 8-87 in Present: Kinetic theory of gases, 1958)
        # (experimental coefficient from Hanson and Eisele is 0.094 +- 0.0012 atm cm^2 s^-1)
        D_acid_N2 = (
            3
            / 8
            / 101325
            * BOLTZMANN_CONSTANT
            * self.temperature
            / ((radius + self.radius_N2) ** 2)
            * math.sqrt(
                BOLTZMANN_CONSTANT
                * self.temperature
                / 2
                / math.pi
                * (1 / mass1 + 1 / self.mass_N2)
            )
        )  # at 1 atm
        wl_term_0 = (
            D_acid_N2 / (self.flow_tube_pressure / 101325) * self.D_to_wl
        )  # s^-1, converted to the given pressure
        D0_factor = (
            1
            / ((radius + self.radius_N2) ** 2)
            * math.sqrt(1 / mass1 + 1 / self.mass_N2)
        )

        rates = np.zeros(n_clusters)
        for i, cluster in enumerate(clusters):
            for n_water, weight_i, label_i in cluster_properties.iter_hydration_states(cluster, conditions.temperature, conditions.partial_pressure_water):
                mass_i, radius_i = cluster.get_hydrated_mass_and_radius(n_water)
                # Average over hydrate distributions
                rates[i] += weight_i * self._wall_loss_rate(radius_i, mass_i, wl_term_0, D0_factor)
        return rates

    def _wall_loss_rate(
        self, cluster_radius: float, cluster_mass: float, wl_term_0: float, D0_factor: float
    ) -> float:
        """Calculate wall loss rate for a single cluster using diffusion."""
        D_factor = (
            1
            / (cluster_radius + self.radius_N2) ** 2
            * math.sqrt(1 / cluster_mass + 1 / self.mass_N2)
        )
        wl_coef = D_factor / (D0_factor * wl_term_0)
        return wl_coef


class Cloud4JAWallLoss(WallLossModel):
    """Cloud 4 wall loss parameterization from Joao Almeida et al., 2013."""
    @property
    def name(self) -> str:
        return WallLossType.CLOUD4_JA.value

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate wall loss coefficients using Cloud4_JA parameterization."""
        n_clusters = len(clusters)
        rates = np.zeros(n_clusters)

        for i, cluster in enumerate(clusters):
            for n_water, weight_i, label_i in cluster_properties.iter_hydration_states(
                cluster,
                conditions.temperature,
                conditions.partial_pressure_water,
            ):
                mass_i, radius_i = cluster.get_hydrated_mass_and_radius(n_water)
                # Average over hydrate distributions
                rates[i] += weight_i * self._wall_loss_rate(radius_i, mass_i)

        return rates

    def _wall_loss_rate(self, cluster_radius: float, cluster_mass: float) -> float:
        """Calculate wall loss rate using Cloud4_JA formula."""
        # From the paper: Molecular understanding of sulphuric acid–amine particle nucleation in the atmosphere
        # https://doi.org/10.1038/nature12663 (see supplementary information)

        # WL factor proportional to 1/mobility diameter
        # Formula: 1.66e-12 / ((d + 0.3e-9) * sqrt(1 + 28.8*u/m))
        # where u is the atomic mass unit (1.66053907e-27 kg)
        # d is the diameter of the cluster in m
        # m is the mass of the cluster in kg
        # 0.3e-9 is the correction factor for the mobility diameter from the paper: An Instrumental Comparison of Mobility and Mass
        # Measurements of Atmospheric Small Ions, DOI: 10.1080/02786826.2010.547890
        # 1.66e-12 is the constant factor from the paper: Molecular understanding of sulphuric acid–amine particle nucleation in the atmosphere
        # https://doi.org/10.1038/nature12663 (see supplementary information)

        u = ureg("1 amu").to("kg").magnitude
        # Mobility diameter calculation
        mobility_diameter = (cluster_radius + 0.3e-9) * math.sqrt(
            1 + (28.8 * u / cluster_mass)
        )
        # Wall loss rate
        rate = 1.66e-12 / mobility_diameter

        return rate


class Cloud4SimpleWallLoss(WallLossModel):
    """Simplified Cloud 4 wall loss parameterization based on the paper by Andreas Kürten et al., 2015"""

    @property
    def name(self) -> str:
        return WallLossType.CLOUD4_SIMPLE.value

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate wall loss coefficients using simplified Cloud4 parameterization."""
        n_clusters = len(clusters)
        rates = np.zeros(n_clusters)

        for i, cluster in enumerate(clusters):
            for n_water, weight_i, label_i in cluster_properties.iter_hydration_states(
                cluster,
                conditions.temperature,
                conditions.partial_pressure_water,
            ):
                mass_i, radius_i = cluster.get_hydrated_mass_and_radius(n_water)
                # Average over hydrate distributions
                rates[i] += weight_i * self._wall_loss_rate(radius_i)

        return rates

    def _wall_loss_rate(self, cluster_radius: float) -> float:
        # WL factor proportional to 1/mobility diameter
        # mobility diameter = mass diameter + 0.3 nm (no mass dependence)
        rate = 1.0e-12 / (2 * cluster_radius + 0.3e-9)
        return rate


class ConstantWallLoss(WallLossModel):
    """Constant wall loss coefficient for all clusters."""

    @property
    def name(self) -> str:
        return self._name

    def __init__(self, rate: float = 0.0):
        """
        Initialize constant wall loss model.

        Args:
            rate: Wall loss coefficient in s⁻¹ (default: 0.0 for no wall losses)
        """
        self.rate = rate
        self._name = f"{WallLossType.CONSTANT.value}_with_rate_{rate}s^-1"

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate constant wall loss coefficients."""
        n_clusters = len(clusters)
        rates = np.full(n_clusters, self.rate)
        return rates


class ExternalWallLoss(WallLossModel):
    """Wall loss coefficients provided externally via dictionary."""

    def __init__(self, rates: Dict[str, float]):
        """
        Initialize external wall loss model.

        Args:
            rates: Dictionary mapping cluster string representations to coefficients (s⁻¹)
        """
        self.rates = rates
        self._name = f"{WallLossType.EXTERNAL.value}_with_rates_{rates}"

    @property
    def name(self) -> str:
        return self._name

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate wall loss coefficients from external dictionary."""
        n_clusters = len(clusters)
        rates = np.zeros(n_clusters)

        for i, cluster in enumerate(clusters):
            cluster_key = str(cluster)
            if cluster_key in self.rates:
                rates[i] = self.rates[cluster_key]
            else:
                raise ValueError(f"Wall loss coefficient not found for cluster {cluster_key}")

        return rates


class WallLossCalculator:
    """Main calculator for wall loss coefficients."""

    def __init__(self, model: Optional[WallLossModel] = None, ion_enhancement_factor: Optional[float] = None):
        """
        Initialize wall loss calculator.

        Args:
            model: Wall loss model to use
            ion_enhancement_factor: If set, multiply wall loss rates for charged clusters by this factor
        """
        self.model = model
        self.ion_enhancement_factor = ion_enhancement_factor

    def set_model(self, model: WallLossModel):
        """Set the wall loss model."""
        self.model = model

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate wall loss coefficients using the current model."""
        if self.model is None:
            logger.warning("No wall loss model set. Returning zero wall loss coefficients.")
            return np.zeros(len(clusters))
        logger.info(f"Calculating wall loss coefficients using {self.model.name}")
        rates = self.model.calculate_coefficients(clusters, cluster_properties, conditions)
        if self.ion_enhancement_factor is not None:
            ions_present = False
            for i, cluster in enumerate(clusters):
                if cluster.charge != 0:
                    ions_present = True
                    rates[i] *= self.ion_enhancement_factor
            if not ions_present:
                logger.debug("No ions present in the cluster collection. Ion enhancement factor will not be applied.")
            else:
                logger.debug(f"Ion enhancement factor {self.ion_enhancement_factor} applied to charged clusters.")
        return rates

    @classmethod
    def create_diffusion(
        cls,
        tube_radius: float,
        temperature: float = 298.15,
        pressure: float = 101325.0,
        ion_enhancement_factor: Optional[float] = None,
    ) -> "WallLossCalculator":
        """Create a wall loss calculator with diffusion model."""
        return cls(
            DiffusionWallLoss(tube_radius, pressure, temperature),
            ion_enhancement_factor=ion_enhancement_factor,
        )

    @classmethod
    def create_cloud4_ja(cls, fwl: float = DEFAULT_FACTOR_FOR_WALL_LOSS_ION_ENHANCEMENT) -> "WallLossCalculator":
        """Create a wall loss calculator with Cloud4_JA model
        Args:
            fwl: Factor for wall loss ion enhancement (default: 3.3)
        Returns:
            WallLossCalculator: Wall loss calculator with Cloud4_JA model and ion enhancement
        """
        return cls(Cloud4JAWallLoss(), ion_enhancement_factor=fwl)

    @classmethod
    def create_cloud4_simple(cls, fwl: float = DEFAULT_FACTOR_FOR_WALL_LOSS_ION_ENHANCEMENT) -> "WallLossCalculator":
        """Create a wall loss calculator with Cloud4_simple model
        Args:
            fwl: Factor for wall loss ion enhancement (default: 3.3)
        Returns:
            WallLossCalculator: Wall loss calculator with Cloud4_simple model and ion enhancement
        """
        return cls(Cloud4SimpleWallLoss(), ion_enhancement_factor=fwl)

    @classmethod
    def create_constant(cls, rate: float = 0.0, fwl: float = DEFAULT_FACTOR_FOR_WALL_LOSS_ION_ENHANCEMENT) -> "WallLossCalculator":
        """Create a wall loss calculator with constant model
        Args:
            rate: Constant wall loss coefficient (default: 0.0)
            fwl: Factor for wall loss ion enhancement (default: 3.3)
        Returns:
            WallLossCalculator: Wall loss calculator with constant model and ion enhancement
        """
        return cls(ConstantWallLoss(rate), ion_enhancement_factor=fwl)

    @classmethod
    def create_external(cls, rates: Dict[str, float], fwl: float = DEFAULT_FACTOR_FOR_WALL_LOSS_ION_ENHANCEMENT) -> "WallLossCalculator":
        """Create a wall loss calculator with external model
        Args:
            rates: Dictionary mapping cluster string representations to coefficients (s⁻¹)
            fwl: Factor for wall loss ion enhancement (default: 3.3)
        Returns:
            WallLossCalculator: Wall loss calculator with external model and ion enhancement
        """
        return cls(ExternalWallLoss(rates), ion_enhancement_factor=fwl)

    def get_supported_types(self) -> List[WallLossType]:
        """Get list of supported wall loss types."""
        return [
            WallLossType.DIFFUSION,
            WallLossType.CLOUD4_JA,
            WallLossType.CLOUD4_SIMPLE,
            WallLossType.CONSTANT,
            WallLossType.EXTERNAL,
        ]
