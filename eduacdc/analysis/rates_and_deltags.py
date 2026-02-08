"""
Analysis functions for rates and DeltaG visualization in ACDC systems.

This module provides functions to analyze and plot rate constants,
free energy surfaces, and related quantities for ACDC cluster systems.
"""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..core.cluster_properties import ClusterProperties
from ..core.clusters import ClusterCollection
from ..core.system import SimulationSystem
from ..utils.constants import (
    BOLTZMANN_CONSTANT,
    ureg,
)


def plot_reference_deltag_surface(
    system: SimulationSystem,
    compound_A: str,
    compound_B: str,
    max_A: int,
    max_B: int,
    charge: int = 0,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    add_diagonal: bool = True,
    output_units: str = "kcal/mol",
    **kwargs,
) -> plt.Axes:
    """
    Plot the standard (reference) DeltaG surface for the given system.

    Parameters
    ----------
    system : SimulationSystem
        The system containing clusters and energies.
    compound_A : str
        Name of compound A (e.g., 'H2SO4').
    compound_B : str
        Name of compound B (e.g., 'NH3').
    max_A : int
        Maximum number of A molecules to include.
    max_B : int
        Maximum number of B molecules to include.
    charge : int, optional
        Charge state to include (0=neutral, -1=neg, +1=pos).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure.
    show : bool, optional
        Whether to show the plot.
    add_diagonal : bool, optional
        Whether to add a diagonal line for nA = nB.
    output_units : str, optional
        Units of the DeltaG values. Default is "kcal/mol".
    **kwargs
        Additional plotting options passed to imshow.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot.
    """

    # Validate inputs
    if charge not in [-1, 0, 1]:
        raise ValueError(f"Invalid charge {charge}. Must be -1, 0, or 1.")

    if max_A < 0 or max_B < 0:
        raise ValueError("max_A and max_B must be non-negative.")

    # Get clusters with the specified charge
    if charge == 0:
        clusters = system.clusters.get_neutral_clusters()
    elif charge == 1:
        clusters = system.clusters.get_positive_clusters()
    else:  # charge == -1
        clusters = system.clusters.get_negative_clusters()

    if not clusters:
        raise ValueError(f"No clusters found with charge {charge}.")

    # Build DeltaG matrix
    deltag_matrix = _build_reference_deltag_matrix(
        system.clusters, system.cluster_properties, compound_A, compound_B, max_A, max_B, system.conditions.temperature
    )

    # results are in SI units
    # Convert to output units
    deltag_matrix = (deltag_matrix * ureg("J/particle")).to(output_units).magnitude
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    plot_kwargs = {
        "cmap": "coolwarm",
        "edgecolors": "black",
        "linewidths": 0.5,
    }
    plot_kwargs.update(kwargs)
    # Plot the DeltaG surface
    im = ax.pcolormesh(
        deltag_matrix.T,  # transpose the matrix to match the plot (A on x-axis, B on y-axis)
        **plot_kwargs,
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(rf"$\Delta G_{{ref}}$ ({output_units})", rotation=270, labelpad=20)

    # Annotate each cell with DeltaG value in kcal (rounded to 2 decimals)
    for nA in range(max_A + 1):
        for nB in range(max_B + 1):
            value = deltag_matrix[nA, nB]
            if not np.isnan(value):
                ax.text(
                    nA + 0.5,
                    nB + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="medium",
                )

    # Add diagonal line if requested
    if add_diagonal:
        max_diag = min(max_A, max_B)
        ax.plot(
            [0, max_diag + 1],
            [0, max_diag + 1],
            alpha=0.5,
            linewidth=3,
            color="lightgray",
        )

    # Set labels and title
    ax.set_xlabel(f"Number of {compound_A} molecules")
    ax.set_ylabel(f"Number of {compound_B} molecules")
    ax.set_title(
        rf"Standard $\Delta G_{{ref}}$ at $T={system.conditions.temperature:.2f}$ K",
        ha="center",
    )

    # Set tick marks at integer positions
    ax.set_xticks(np.arange(0, max_B + 1) + 0.5)
    ax.set_yticks(np.arange(0, max_A + 1) + 0.5)
    ax.set_xticklabels(np.arange(0, max_B + 1))
    ax.set_yticklabels(np.arange(0, max_A + 1))

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def _build_reference_deltag_matrix(
    clusters: ClusterCollection,
    cluster_properties: ClusterProperties,
    compound_A: str,
    compound_B: str,
    max_A: int,
    max_B: int,
    temperature: float,
) -> np.ndarray:
    """
    Build the reference DeltaG matrix for the given clusters.

    Parameters
    ----------
    clusters : List[Cluster]
        List of clusters to analyze.
    compound_A : str
        Symbol for compound A.
    compound_B : str
        Symbol for compound B.
    max_A : int
        Maximum number of A molecules.
    max_B : int
        Maximum number of B molecules.
    temperature : float
        Temperature of the system.
    Returns
    -------
    np.ndarray
        Matrix of DeltaG values, shape (max_A + 1, max_B + 1).
        NaN values for combinations that don't exist.
    """
    # Initialize matrix with NaN
    deltag_matrix = np.full((max_A + 1, max_B + 1), np.nan)

    # Create lookup dictionary for clusters
    cluster_lookup = {}
    for cluster in clusters:
        nA = cluster.get_molecule_count(compound_A)
        nB = cluster.get_molecule_count(compound_B)
        cluster_lookup[(nA, nB)] = cluster

    # Calculate DeltaG for each combination
    for nA in range(max_A + 1):
        for nB in range(max_B + 1):
            if (nA, nB) in cluster_lookup:
                cluster = cluster_lookup[(nA, nB)]
                # access the cluster gibbs free energy
                deltag = cluster_properties.get_reference_gibbs_free_energy(cluster.label, temperature)
                deltag_matrix[nA, nB] = deltag

    return deltag_matrix


def get_monomer_vapor_pressures(
    monomer_concentrations: Dict[str, str],
    temperature: float,
) -> Dict[str, float]:
    """Get the monomer vapor pressures for the given compounds.
    The monomer vapor pressures are calculated using the ideal gas law.
    The unit of the monomer concentrations is handled by pint.
    The unit of the monomer vapor pressures is Pa.

    Args:
        monomer_concentrations: Dictionary of monomer concentrations. Can be:
            - Dict[str, str]: Values with units as strings (e.g., "1e6 cm^-3", "1 ppt")
        temperature (float): Temperature of the system in K.

    Raises:
        ValueError: If the unit is invalid.

    Returns:
        Dict[str, float]: Dictionary of monomer vapor pressures in Pa.
    """
    monomer_vapor_pressures = {}
    # assume that air is ideal and 1 atm pressure
    reference_pressure = ureg("1atm")
    temperature = temperature * ureg.kelvin
    with ureg.context("conc", temperature=temperature, pressure=reference_pressure):
        for monomer, conc in monomer_concentrations.items():
            if isinstance(conc, str):
                # convert to Pa
                vapor_pressure = ureg(conc).to("Pa").magnitude
            else:
                raise ValueError(f"Invalid concentration format for {monomer}: {conc}. Specify concentration as a string with units (e.g., '1e6 cm^-3' or '140 ppt').")
            monomer_vapor_pressures[monomer] = vapor_pressure
    return monomer_vapor_pressures


def _build_act_deltag_matrix(
    clusters: ClusterCollection,
    cluster_properties: ClusterProperties,
    compound_A: str,
    compound_B: str,
    max_A: int,
    max_B: int,
    monomer_vapor_pressure_A: float,  # Pa
    monomer_vapor_pressure_B: float,  # Pa
    temperature: float,  # K
) -> np.ndarray:
    """Build the actual DeltaG matrix for the given clusters.

    Args:
        clusters (List[Cluster]): List of clusters to analyze.
        compound_A (str): Symbol for compound A.
        compound_B (str): Symbol for compound B.
        max_A (int): Maximum number of A molecules.
        max_B (int): Maximum number of B molecules.
        monomer_vapor_pressure_A (float): Vapor pressure of compound A in Pa.
        monomer_vapor_pressure_B (float): Vapor pressure of compound B in Pa.
        temperature (float): Temperature of the system in K.

    Returns:
        np.ndarray: Matrix of actual DeltaG values, shape (max_A + 1, max_B + 1).
        NaN values for combinations that don't exist.
    """
    # Initialize matrix with NaN
    deltag_matrix = np.full((max_A + 1, max_B + 1), np.nan)

    # Create lookup dictionary for clusters
    cluster_lookup = {}
    for cluster in clusters:
        nA = cluster.get_molecule_count(compound_A)
        nB = cluster.get_molecule_count(compound_B)
        cluster_lookup[(nA, nB)] = cluster

    monomer_vapor_pressures = {
        compound_A: monomer_vapor_pressure_A,
        compound_B: monomer_vapor_pressure_B,
    }
    # Calculate DeltaG for each combination
    for nA in range(max_A + 1):
        for nB in range(max_B + 1):
            if (nA, nB) in cluster_lookup:
                cluster = cluster_lookup[(nA, nB)]
                # access the cluster gibbs free energy
                deltag_act = cluster_properties.get_actual_gibbs_free_energy(cluster, temperature, monomer_vapor_pressures)
                deltag_matrix[nA, nB] = deltag_act

    return deltag_matrix


def plot_act_deltag_surface(
    system: SimulationSystem,
    compound_A: str,
    compound_B: str,
    max_A: int,
    max_B: int,
    monomer_concentrations: Dict[str, str],
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    add_diagonal: bool = True,
    output_units: str = "kcal/mol",
    **kwargs,
) -> plt.Axes:
    """Plot the actual DeltaG surface for the given system.

    Args:
        system (SimulationSystem): The system containing clusters and energies.
        compound_A (str): Symbol for compound A.
        compound_B (str): Symbol for compound B.
        max_A (int): Maximum number of A molecules.
        max_B (int): Maximum number of B molecules.
        monomer_concentrations(Dict[str, str]): Dictionary of monomer concentrations. Values with units as strings (e.g., "1e6 cm^-3", "1 ppt")
        ax (Optional[plt.Axes]): Axis to plot on. If None, creates a new figure.
        show (bool): Whether to show the plot.
        add_diagonal (bool): Whether to add a diagonal line for nA = nB.
        output_units (str): Units of the DeltaG values. Default is "kcal/mol".
        **kwargs: Additional plotting options passed to imshow.

    Returns:
        plt.Axes: The axis with the plot.
    """
    monomer_vapor_pressures = get_monomer_vapor_pressures(
        monomer_concentrations, system.conditions.temperature
    )
    deltag_matrix = _build_act_deltag_matrix(
        system.clusters, system.cluster_properties, 
        compound_A,
        compound_B,
        max_A,
        max_B,
        monomer_vapor_pressures[compound_A],
        monomer_vapor_pressures[compound_B],
        system.conditions.temperature,
    )
    # results are in SI units (J/particle)
    # convert to output units
    deltag_matrix = (deltag_matrix * ureg("J/particle")).to(output_units).magnitude
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()
    plot_kwargs = {
        "cmap": "coolwarm",
        "edgecolors": "black",
        "linewidths": 0.5,
    }
    plot_kwargs.update(kwargs)
    # Plot the DeltaG surface
    im = ax.pcolormesh(
        deltag_matrix.T,  # transpose the matrix to match the plot (A on x-axis, B on y-axis)
        **plot_kwargs,
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        rf"Actual $\Delta G$ ({output_units})", rotation=270, labelpad=20, ha="center"
    )

    # Annotate each cell with DeltaG value in kcal (rounded to 2 decimals)
    for nA in range(max_A + 1):
        for nB in range(max_B + 1):
            value = deltag_matrix[nA, nB]
            if not np.isnan(value):
                ax.text(
                    nA + 0.5,
                    nB + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="medium",
                )

    # Add diagonal line if requested
    if add_diagonal:
        max_diag = min(max_A, max_B)
        ax.plot(
            [0, max_diag + 1],
            [0, max_diag + 1],
            alpha=0.5,
            linewidth=3,
            color="lightgray",
        )

    # Format concentration strings for display
    def format_concentration(conc):
        if isinstance(conc, tuple):
            return f"{conc[0]:g} {conc[1]}"
        elif isinstance(conc, (int, float)):
            return f"{conc:g} m^-3"
        elif isinstance(conc, str):
            value = ureg(conc)
            return f"{value.magnitude:g} {value.units:~P}"
        else:
            return str(conc)

    ax.set_title(
        rf"Actual $\Delta G$ at $T={system.conditions.temperature:g}$ K"
        "\n"
        rf"[{compound_A}] = {format_concentration(monomer_concentrations[compound_A])}"
        " "
        rf"[{compound_B}] = {format_concentration(monomer_concentrations[compound_B])}",
        ha="center",
        fontsize="large",
    )
    # Set labels
    ax.set_xlabel(f"Number of {compound_A} molecules", fontsize="large")
    ax.set_ylabel(f"Number of {compound_B} molecules", fontsize="large")

    # Set tick marks at integer positions
    ax.set_xticks(np.arange(0, max_B + 1) + 0.5)
    ax.set_yticks(np.arange(0, max_A + 1) + 0.5)
    ax.set_xticklabels(np.arange(0, max_B + 1))
    ax.set_yticklabels(np.arange(0, max_A + 1))
    if show:
        plt.tight_layout()
        plt.show()

    return ax


def _build_overall_evaporation_rate_matrix(
    system: SimulationSystem,
    compound_A: str,
    compound_B: str,
    max_A: int,
    max_B: int,
) -> np.ndarray:
    """Build the overall evaporation rate matrix for the given system."""
    evaporation_coefficients = system.get_evaporation_coefficients()
    evaporation_processes = system.get_evaporation_processes()
    # build the matrix
    overall_evaporation_rate_matrix = np.full((max_A + 1, max_B + 1), np.nan)
    # Create lookup dictionary for clusters

    # Calculate overall evaporation rate for each combination
    for nA in range(max_A + 1):
        for nB in range(max_B + 1):
            composition = {compound_A: nA, compound_B: nB}
            cluster_index = system.clusters.get_index_by_composition(composition)
            E_vals = evaporation_coefficients[cluster_index]
            if cluster_index not in evaporation_processes:
                continue
            else:
                overall_evaporation_rate_matrix[nA, nB] = 0
            reaction_indices = evaporation_processes[cluster_index]
            for i, j in reaction_indices:
                # i and j are the indices of the clusters that are formed
                # we need to add the evaporation rate of the cluster that evaporates to the overall evaporation rate
                overall_evaporation_rate_matrix[nA, nB] += E_vals[i]

    return overall_evaporation_rate_matrix


def plot_overall_evaporation_rate_surface(
    system: SimulationSystem,
    compound_A: str,
    compound_B: str,
    max_A: int,
    max_B: int,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    add_diagonal: bool = False,
    output_units: str = "1/s",
    **kwargs,
) -> plt.Axes:
    """Plot the overall evaporation rate surface for the given system."""
    overall_evaporation_rate_matrix = _build_overall_evaporation_rate_matrix(
        system, compound_A, compound_B, max_A, max_B
    )
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()
    plot_kwargs = {
        "cmap": "coolwarm",
        "edgecolors": "black",
        "linewidths": 0.5,
        "norm": "log",
    }
    plot_kwargs.update(kwargs)
    # Plot the overall evaporation rate surface
    im = ax.pcolormesh(
        overall_evaporation_rate_matrix.T,  # transpose the matrix to match the plot (A on x-axis, B on y-axis)
        **plot_kwargs,
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        rf"Overall Evaporation Rate ({output_units})",
        rotation=270,
        labelpad=20,
        ha="center",
    )

    # Add diagonal line if requested
    if add_diagonal:
        max_diag = min(max_A, max_B)
        ax.plot(
            [0, max_diag + 1],
            [0, max_diag + 1],
            alpha=0.5,
            linewidth=3,
            color="lightgray",
        )
        # Annotate each cell with DeltaG value in kcal (rounded to 2 decimals)
    for nA in range(max_A + 1):
        for nB in range(max_B + 1):
            value = overall_evaporation_rate_matrix[nA, nB]
            if not np.isnan(value):
                ax.text(
                    nA + 0.5,
                    nB + 0.5,
                    f"{value:.1g}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="medium",
                )
        # Set labels
    ax.set_xlabel(f"Number of {compound_A} molecules", fontsize="large")
    ax.set_ylabel(f"Number of {compound_B} molecules", fontsize="large")

    # Set tick marks at integer positions
    ax.set_xticks(np.arange(0, max_B + 1) + 0.5)
    ax.set_yticks(np.arange(0, max_A + 1) + 0.5)
    ax.set_xticklabels(np.arange(0, max_B + 1))
    ax.set_yticklabels(np.arange(0, max_A + 1))
    ax.set_title(
        rf"Overall Evaporation Rate at $T={system.conditions.temperature:g}$ K",
    )
    if show:
        plt.tight_layout()
        plt.show()

    return ax


def _build_monomer_collision_rate_matrix(
    system: SimulationSystem,
    compound_A: str,
    compound_B: str,
    max_A: int,
    max_B: int,
    monomer_vapor_pressure_A: float,  # Pa
    monomer_vapor_pressure_B: float,  # Pa
) -> np.ndarray:
    """Build the monomer collision rate matrix for the given system."""
    collision_coefficients = system.get_collision_coefficients()
    monomer_A_collision_rate_matrix = np.full((max_A + 1, max_B + 1), np.nan)
    monomer_B_collision_rate_matrix = np.full((max_A + 1, max_B + 1), np.nan)
    monomer_A_index = system.clusters.get_index_by_label(f"1{compound_A}")
    monomer_B_index = system.clusters.get_index_by_label(f"1{compound_B}")
    temperature = system.conditions.temperature
    # build the matrix
    for nA in range(max_A + 1):
        for nB in range(max_B + 1):
            composition = {compound_A: nA, compound_B: nB}
            cluster_index = system.clusters.get_index_by_composition(composition)
            if cluster_index is None:
                continue
            # get the collision rate of the cluster with the monomer A and B
            coll_cluster_A = collision_coefficients[cluster_index][monomer_A_index]
            coll_cluster_B = collision_coefficients[cluster_index][monomer_B_index]
            monomer_A_collision_rate_matrix[nA, nB] = (
                coll_cluster_A
                * monomer_vapor_pressure_A
                / (BOLTZMANN_CONSTANT * temperature)
            )
            monomer_B_collision_rate_matrix[nA, nB] = (
                coll_cluster_B
                * monomer_vapor_pressure_B
                / (BOLTZMANN_CONSTANT * temperature)
            )
    return monomer_A_collision_rate_matrix, monomer_B_collision_rate_matrix


def plot_monomer_collision_rate_ratio(
    system: SimulationSystem,
    compound_A: str,
    compound_B: str,
    max_A: int,
    max_B: int,
    monomer_concentrations: Dict[str, str],
    fig: Optional[plt.Figure] = None,
    show: bool = False,
    output_units: str = "1/s",
    **kwargs,
) -> plt.Axes:
    """Plot the monomer collision rate surface for the given system."""
    monomer_vapor_pressures = get_monomer_vapor_pressures(
        monomer_concentrations, system.conditions.temperature
    )
    monomer_vapor_pressure_A = monomer_vapor_pressures[compound_A]
    monomer_vapor_pressure_B = monomer_vapor_pressures[compound_B]
    monomer_A_collision_rate_matrix, monomer_B_collision_rate_matrix = (
        _build_monomer_collision_rate_matrix(
            system,
            compound_A,
            compound_B,
            max_A,
            max_B,
            monomer_vapor_pressure_A,
            monomer_vapor_pressure_B,
        )
    )
    # Create plot
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
    total_evaporation_rate_matrix = _build_overall_evaporation_rate_matrix(
        system, compound_A, compound_B, max_A, max_B
    )
    mat_A = monomer_A_collision_rate_matrix / total_evaporation_rate_matrix
    mat_B = monomer_B_collision_rate_matrix / total_evaporation_rate_matrix
    plot_kwargs = {
        "cmap": "coolwarm",
        "edgecolors": "black",
        "linewidths": 0.5,
        "norm": "log",
    }
    plot_kwargs.update(kwargs)
    # Plot the monomer collision rate surface
    im = ax1.pcolormesh(
        mat_A.T,  # transpose the matrix to match the plot (A on x-axis, B on y-axis)
        **plot_kwargs,
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label(
        r"$({\beta_A}C_A)/\sum \gamma$", rotation=270, labelpad=20, ha="center"
    )

    im = ax2.pcolormesh(
        mat_B.T,  # transpose the matrix to match the plot (A on x-axis, B on y-axis)
        **plot_kwargs,
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label(
        r"$({\beta_B}C_B)/\sum \gamma$", rotation=270, labelpad=20, ha="center"
    )

    ax1.set_title(
        rf"collision rate of A/ total evaporation rate at [{compound_A}] = {monomer_concentrations[compound_A]}"
    )
    ax2.set_title(
        rf"collision rate of B/ total evaporation rate at [{compound_B}] = {monomer_concentrations[compound_B]}"
    )
    fig.suptitle(
        rf"at $T={system.conditions.temperature:g}$ K"
    )

    # Annotate each cell with DeltaG value in kcal (rounded to 2 decimals)
    for nA in range(max_A + 1):
        for nB in range(max_B + 1):
            value = mat_A[nA, nB]
            if not np.isnan(value):
                ax1.text(
                    nA + 0.5,
                    nB + 0.5,
                    f"{value:.1g}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="medium",
                )
            value = mat_B[nA, nB]
            if not np.isnan(value):
                ax2.text(
                    nA + 0.5,
                    nB + 0.5,
                    f"{value:.1g}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="medium",
                )
    for _ax in [ax1, ax2]:  # Set tick marks at integer positions
        _ax.set_xticks(np.arange(0, max_B + 1) + 0.5)
        _ax.set_yticks(np.arange(0, max_A + 1) + 0.5)
        _ax.set_xticklabels(np.arange(0, max_B + 1))
        _ax.set_yticklabels(np.arange(0, max_A + 1))
        _ax.set_xlabel(f"Number of {compound_A} molecules", fontsize="large")
        _ax.set_ylabel(f"Number of {compound_B} molecules", fontsize="large")
    if show:
        plt.tight_layout()
        plt.show()
    return ax1, ax2
