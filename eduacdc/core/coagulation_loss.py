"""
Coagulation loss coefficient parameterization.
"""

import logging
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

import numpy as np

from ..utils.constants import AVOGADRO_CONSTANT, BOLTZMANN_CONSTANT, PI
from .cluster_properties import ClusterProperties
from .clusters import ClusterCollection
from .system import AmbientConditions

logger = logging.getLogger(__name__)

DEFAULT_COAGULATION_COEFFICIENT = 1e-3  # s⁻¹, corresponding to average boundary-layer conditions
DEFAULT_COAGULATION_EXPONENT = -1.6 # determined for a background distribution at d_bg = 100 nm
DEFAULT_BACKGROUND_CONCENTRATION = 1e3  # cm⁻³, corresponding to average boundary-layer conditions
DEFAULT_BACKGROUND_DIAMETER = 100  # nm, corresponding to average boundary-layer conditions
DEFAULT_BACKGROUND_DENSITY = 1000  # kg/m³, corresponding to an average value determined for larger particles
DEFAULT_REFERENCE_CLUSTER = "1A" # (assumed to refer to sulfuric acid

class CoagulationLossType(Enum):
    """Types of coagulation loss parameterizations."""

    EXPONENTIAL = "exp_loss"
    BACKGROUND = "bg_loss"
    CONSTANT = "constant"


class CoagulationLossModel(ABC):
    """Abstract base class for coagulation calculation models."""
    
    @property
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Get the name of the coagulation model."""
        pass
    
    @abstractmethod
    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate coagulation loss coefficients for all clusters."""
        pass
    

class ExponentialCoagulationLoss(CoagulationLossModel):
    """Exponential coagulation loss parameterization.
    
    Implements size-dependent coagulation loss following:
    CS(i) = A * (r_i / r_ref)^B
    
    Where:
    - A = coefficient (default: 1e-3 s⁻¹)
    - B = exponent (default: -1.6)
    - r_i = radius of cluster i
    - r_ref = reference cluster radius
    """
    
    @property
    def name(self) -> str:
        return CoagulationLossType.EXPONENTIAL.value

    def __init__(
        self,
        coefficient: float = DEFAULT_COAGULATION_COEFFICIENT,
        exponent: float = DEFAULT_COAGULATION_EXPONENT,
        reference_radius: Optional[float] = None,
        reference_cluster: Optional[str] = None,
    ):
        self.coefficient = coefficient
        self.exponent = exponent
        self.reference_radius = reference_radius
        self.reference_cluster = reference_cluster
        self._cached_reference_radius: Optional[float] = None

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate exponential coagulation loss coefficients."""
        rates = np.zeros(len(clusters))
        
        # Determine reference radius
        r_ref = self._get_reference_radius(clusters)
        
        if r_ref <= 0:
            logger.warning("Reference radius is zero or negative, returning zero rates")
            return rates
        
        for i, cluster in enumerate(clusters):
            for n_water, weight_i, label_i in cluster_properties.iter_hydration_states(cluster, conditions.temperature, conditions.partial_pressure_water):
                mass_i, radius_i = cluster.get_hydrated_mass_and_radius(n_water)
                rates[i] += weight_i * self._coag_coefficient(radius_i, r_ref)
        
        return rates

    def _coag_coefficient(self, cluster_radius: float, reference_radius: float) -> float:
        """Calculate the coagulation coefficient for a cluster."""
        return self.coefficient * (cluster_radius / reference_radius) ** self.exponent
    
    def _get_reference_radius(self, clusters: ClusterCollection) -> float:
        """Get the reference radius for scaling."""
        if self._cached_reference_radius is not None:
            return self._cached_reference_radius
        
        if self.reference_radius is not None:
            r_ref = self.reference_radius
        elif self.reference_cluster is not None:
            r_ref = self._get_cluster_radius(clusters, self.reference_cluster)
        else:
            r_ref = self._get_monomer_radius(clusters)
        
        self._cached_reference_radius = r_ref
        return r_ref

    def _get_cluster_radius(self, clusters: ClusterCollection, cluster_label: str) -> float:
        """Get radius of a specific cluster."""
        try:
            cluster = clusters.get_by_label(cluster_label)
            logger.debug(f"Reference cluster {cluster_label} found and using radius = {cluster.radius:g} for calculating coagulation losses.")
        except KeyError:
            raise ValueError(f"Reference cluster {cluster_label} not found for calculating reference coagulation losses")
        return cluster.radius

    def _get_monomer_radius(self, clusters: ClusterCollection) -> float:
        """Get radius of the monomer (first cluster)."""
        try:
            cluster = clusters.get_by_label(DEFAULT_REFERENCE_CLUSTER)
            logger.debug(f"Reference cluster {DEFAULT_REFERENCE_CLUSTER} found and using radius = {cluster.radius:g} for calculating coagulation losses.")
        except KeyError:
            raise ValueError(f"Reference cluster {DEFAULT_REFERENCE_CLUSTER} not found for calculating reference coagulation losses")
        return cluster.radius


class BackgroundCoagulationLoss(CoagulationLossModel):
    """Background particle coagulation loss parameterization.
    
    Implements collision-based coagulation loss with background particles:
    CS(i) = 4π(r_i + r_bg)(D_i + D_bg) * (1 + Kn)/(1 + 2Kn(1 + Kn)) * c_bg
    
    Uses Dahneke (1983) parameterization for collision coefficients.
    """
    
    @property
    def name(self) -> str:
        return CoagulationLossType.BACKGROUND.value

    def __init__(
        self,
        concentration: float = DEFAULT_BACKGROUND_CONCENTRATION,  # cm⁻³
        diameter: float = DEFAULT_BACKGROUND_DIAMETER,  # nm
        density: float = DEFAULT_BACKGROUND_DENSITY,  # kg/m³
        temperature: float = 298.15, # K
    ):
        self.concentration = concentration * 1e6  # Convert cm⁻³ to m⁻³
        self.radius = diameter * 0.5 * 1e-9  # Convert nm to m (radius)
        self.density = density
        self.temperature = temperature
        self._calculate_background_properties()

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate background coagulation loss coefficients using Dahneke parameterization."""
        # Update temperature and recalculate background properties
        self.temperature = conditions.temperature
        self._calculate_background_properties()
        rates = np.zeros(len(clusters))

                
        for i, cluster in enumerate(clusters):
            for n_water, weight_i, label_i in cluster_properties.iter_hydration_states(cluster, conditions.temperature, conditions.partial_pressure_water):
                mass_i, radius_i = cluster.get_hydrated_mass_and_radius(n_water)
                collision_coeff = self._compute_collision_coefficient(radius_i, mass_i)
                rates[i] += weight_i * collision_coeff * self.concentration
                       
        return rates

    def _compute_collision_coefficient(self, cluster_radius: float, cluster_mass: float) -> float:
        # Calculate cluster diffusion coefficient
        D_i = self._calculate_diffusion_coefficient(
            cluster_radius, cluster_mass
        )

        # Calculate cluster thermal velocity
        c1 = math.sqrt(self.thermal_velocity_coefficient/cluster_mass)
        # Calculate Knudsen number for collision coefficient. 
        Kn = 2*(D_i + self.D_bg)/(math.sqrt(c1**2 + self.c_bg**2)*(cluster_radius + self.radius))
        
        # Dahneke collision coefficient
        collision_coeff = (
            4 * PI * (cluster_radius + self.radius) * 
            (D_i + self.D_bg) * 
            (1 + Kn) / (1 + 2 * Kn * (1 + Kn))
        )
        return collision_coeff

    def _calculate_background_properties(self) -> None:
        """Calculate background particle properties."""

        self.mass = self.density * (4/3) * PI * self.radius**3                        
        # Air viscosity from DMAN (same as Perl code)
        self.air_viscosity = 2.5277e-7 * self.temperature ** 0.75302
        
        # Calculate mean free path of air, assuming standard pressure
        self.mean_free_path = 2*self.air_viscosity/(1.01325e5*math.sqrt(8*0.0289/(PI*AVOGADRO_CONSTANT*BOLTZMANN_CONSTANT*self.temperature)))
        # Atmospheric Chemistry And Physics 2nd Ed J. Seinfeld, S. Pandis
        # Eq. 9.73
        self.diffusion_coefficient = BOLTZMANN_CONSTANT*self.temperature/(3*PI*self.air_viscosity)
        # Coefficient for calculating thermal velocity
        self.thermal_velocity_coefficient = 8*BOLTZMANN_CONSTANT*self.temperature/PI
        # Background particle diffusion coefficient
        self.D_bg = self._calculate_diffusion_coefficient(
            self.radius, self.mass
        )
        # Background particle thermal velocity
        self.c_bg = math.sqrt(self.thermal_velocity_coefficient/self.mass)
        

    def _calculate_diffusion_coefficient(
        self, radius: float, mass: float
    ) -> float:
        """Calculate diffusion coefficient using Dahneke parameterization."""
        # Dahneke diffusion coefficient
        _temp = self.mean_free_path / radius
        D = (self.diffusion_coefficient / (2 * radius)) * (
            (5 + 4 * _temp + 6 * _temp ** 2 + 18 * _temp ** 3) /
            (5 - _temp + (8 + PI) * _temp ** 2)
        )
        
        return D

class ConstantCoagulationLoss(CoagulationLossModel):
    """Constant coagulation loss coefficient for all clusters."""

    @property
    def name(self) -> str:
        return CoagulationLossType.CONSTANT.value

    def __init__(self, rate: float = 0.0):
        self.rate = rate

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Return constant coefficient for all clusters."""
        return np.full(len(clusters), self.rate)


class CoagulationLossCalculator:
    """Main calculator for coagulation loss coefficients."""

    def __init__(self, model: Optional[CoagulationLossModel] = None):
        self.model = model

    def set_model(self, model: CoagulationLossModel):
        """Set the coagulation loss model."""
        self.model = model

    def calculate_coefficients(self, clusters: ClusterCollection, cluster_properties: ClusterProperties, conditions: AmbientConditions) -> np.ndarray:
        """Calculate coagulation loss coefficients for all clusters."""
        if self.model is None:
            return np.zeros(len(clusters))
        logger.info(f"Calculating coagulation loss coefficients using {self.model.name}")
        return self.model.calculate_coefficients(clusters, cluster_properties, conditions)

    @classmethod
    def create_exponential(
        cls,
        coefficient: float = DEFAULT_COAGULATION_COEFFICIENT,
        exponent: float = DEFAULT_COAGULATION_EXPONENT,
        reference_radius: Optional[float] = None,
        reference_cluster: Optional[str] = None,
    ) -> "CoagulationLossCalculator":
        """Create exponential coagulation calculator."""
        return cls(
            ExponentialCoagulationLoss(coefficient, exponent, reference_radius, reference_cluster)
        )

    @classmethod
    def create_background(
        cls,
        concentration: float = DEFAULT_BACKGROUND_CONCENTRATION,
        diameter: float = DEFAULT_BACKGROUND_DIAMETER,
        density: float = DEFAULT_BACKGROUND_DENSITY,
    ) -> "CoagulationLossCalculator":
        """Create background coagulation loss calculator."""
        return cls(BackgroundCoagulationLoss(concentration, diameter, density))

    @classmethod
    def create_constant(cls, rate: float = 0.0) -> "CoagulationLossCalculator":
        """Create constant coagulation calculator."""
        return cls(ConstantCoagulationLoss(rate))

    def get_supported_types(self) -> List[CoagulationLossType]:
        """Get list of supported coagulation types."""
        return list(CoagulationLossType)
