"""
Visualization functions for ACDC simulation results.

This module provides plotting and visualization functions for ACDC simulation results,
including concentration plots, outgrowth rates, and cluster size distributions.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ..simulation.results import SimulationResults
from ..utils import ureg


def plot_final_concentrations(
    results: SimulationResults,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    matlab_cluster_labels: Optional[List[str]] = None,
    output_units: str = "m^-3",
) -> Axes:
    final_concentrations = results.get_final_concentrations(output_units)
    labels = results.system.clusters.get_labels()
    if matlab_cluster_labels:
        y = np.zeros_like(final_concentrations)
        for concentration, label in zip(final_concentrations, labels):
            _idx = matlab_cluster_labels.index(label)
            y[_idx] = concentration
        final_concentrations = y
        labels = matlab_cluster_labels
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y=labels,width=final_concentrations)
    ax.set_xscale("log")
    ax.set_xlabel(f"Concentration ({output_units})")
    ax.set_ylabel("Cluster")
    ax.set_title("Final Concentrations")
    ax.grid(True,alpha=0.3)
    return ax

def plot_concentrations(
    results: SimulationResults,
    ax: Optional[Axes] = None,
    cluster_indices: Optional[List[int]] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    output_units: str = "m^-3",
) -> Axes:
    """
    Plot cluster concentrations over time.

    Parameters
    ----------
    results : SimulationResults
        The simulation results to plot.
    ax : Optional[Axes]
        Existing axes to plot on. If None, creates new figure and axes.
    cluster_indices : Optional[List[int]]
        Indices of clusters to plot. If None, plots first 10 clusters.
    log_scale : bool
        Whether to use log scale for y-axis.
    figsize : tuple
        Figure size (only used if ax is None).

    Returns
    -------
    Axes
        The matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if cluster_indices is None:
        cluster_indices = list(range(min(10, results.system.n_clusters)))

    color_palette = plt.get_cmap("tab20").colors
    colors = [color_palette[i] for i in cluster_indices]
    for i, cluster_index in enumerate(cluster_indices):
        if i < results.system.n_clusters:
            cluster = results.system.clusters[i]
            conc = (results.concentrations[:, i] * ureg("m^-3")).to(output_units).magnitude
            non_zero = conc > 0
            ax.plot(
                results.time[non_zero],
                conc[non_zero],
                label=f"{str(cluster)} ({cluster.type.value})",
                color=colors[i],
            )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Concentration ({ureg(output_units).units:~P})")
    ax.set_title("Cluster Concentrations Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_total_concentrations(
    results: SimulationResults,
    ax: Optional[Axes] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> Axes:
    """
    Plot total concentrations by cluster type.

    Parameters
    ----------
    results : SimulationResults
        The simulation results to plot.
    ax : Optional[Axes]
        Existing axes to plot on. If None, creates new figure and axes.
    log_scale : bool
        Whether to use log scale for y-axis.
    figsize : tuple
        Figure size (only used if ax is None).

    Returns
    -------
    Axes
        The matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(results.time, results.total_neutral, label="Neutral clusters")
    ax.plot(results.time, results.total_positive, label="Positive clusters")
    ax.plot(results.time, results.total_negative, label="Negative clusters")

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total Concentration (m⁻³)")
    ax.set_title("Total Concentrations by Type")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_cluster_size_distribution(
    results: SimulationResults,
    ax: Optional[Axes] = None,
    time_index: int = -1,
    figsize: Tuple[int, int] = (10, 6),
) -> Axes:
    """
    Plot cluster size distribution at a specific time.

    Parameters
    ----------
    results : SimulationResults
        The simulation results to plot.
    ax : Optional[Axes]
        Existing axes to plot on. If None, creates new figure and axes.
    time_index : int
        Time index to plot distribution for. Default is -1 (final time).
    figsize : tuple
        Figure size (only used if ax is None).

    Returns
    -------
    Axes
        The matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sizes = [cluster.total_molecules for cluster in results.system.clusters]
    concentrations = results.concentrations[time_index, :]

    # Group by size
    size_counts = {}
    for size, conc in zip(sizes, concentrations):
        if size not in size_counts:
            size_counts[size] = 0
        size_counts[size] += conc

    sizes_list = sorted(size_counts.keys())
    total_concentrations = [size_counts[size] for size in sizes_list]

    ax.bar(sizes_list, total_concentrations)
    ax.set_xlabel("Cluster Size (molecules)")
    ax.set_ylabel("Total Concentration (m⁻³)")
    ax.set_title(f"Cluster Size Distribution at t = {results.time[time_index]:.1f} s")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    return ax


def plot_formation_rates(
    results: SimulationResults,
    ax: Optional[Axes] = None,
    log_scale: bool = True,
    output_units: str = "cm^-3 s^-1",
    figsize: Tuple[int, int] = (10, 6),
    filter_negative: bool = False,
) -> Axes:
    """
    Plot formation rates over time.

    Parameters
    ----------
    results : SimulationResults
        The simulation results to plot.
    ax : Optional[Axes]
        Existing axes to plot on. If None, creates new figure and axes.
    log_scale : bool
        Whether to use log-log scale for axes.
    figsize : tuple
        Figure size (only used if ax is None).
        filter_negative : bool
        If True, negative outgrowth rates are set to NaN (not plotted).
    output_units : str
        Units of the outgrowth rates. Default is "cm^-3 s^-1".
    Returns
    -------
    Axes
        The matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    y = results.get_formation_rates(output_units)
    time = results.time
    if filter_negative:
        mask = y > 0
        y = y[mask]
        time = time[mask]

    ax.plot(time, y, label="Formation", linewidth=2)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Formation Rate ({output_units})")
    ax.set_title("Formation Rates Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
