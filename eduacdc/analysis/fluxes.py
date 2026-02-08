import logging
import math
from collections import OrderedDict
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes

from eduacdc.core.clusters import ClusterType
from eduacdc.core.equations import ClusterEquations, FluxDirection
from eduacdc.core.system import SimulationSystem
from eduacdc.simulation.results import SimulationResults
from eduacdc.utils import ureg

logger = logging.getLogger(__name__)

def get_significant_outflux_reactions(
    results: SimulationResults,
    threshold: float = 0.01,
    charge_type: str = "all"
) -> tuple[list[tuple[str, str, float]], float]:
    """Get the outgrowth reactions that result in a significant outflux and the total outflux.

    Args:
        results (SimulationResults): Simulation results.
        threshold (float, optional): Threshold for significant outflux. Defaults to 0.01.
        charge_type (str, optional): Charge type to consider. Defaults to "all". Can be "all", "neutral", "positive", or "negative".
    Raises:
        ValueError: If final outflux rates are not available.

    Returns:
        tuple[list[tuple[str, str, float]], float]: List of cluster collision pairs that result in a significant outflux and the total outflux.
    """
    if charge_type not in ["all", "neutral", "positive", "negative"]:
        raise ValueError(
            f"Invalid charge type: {charge_type}. Must be 'all', 'neutral', 'positive', or 'negative'."
        )
    if results.final_outflux_rates is None:
        raise ValueError("Final outflux rates are not available. Please run simulation with `save_final_outflux_rates=True`.")
    
    outflux_rates = results.final_outflux_rates
    reactions = []
    # go through all outgrowth reactions
    if charge_type != "all":
        _charge_type = (
            ClusterType.NEUTRAL
            if charge_type == "neutral"
            else ClusterType.POSITIVE
            if charge_type == "positive"
            else ClusterType.NEGATIVE
        )
    else:
        _charge_type = None
    total_flux_out = 0
    for collision in results.equations.process_tracker.outgrowth_collisions.values():
        r1_idx, r2_idx = collision.reactant_indices
        r1 = results.system.clusters[r1_idx]
        r2 = results.system.clusters[r2_idx]
        valid_reaction = False
        if charge_type != "all":
            # Ensure at least one of the colliders has the required charge type
            if r1.type == _charge_type or r2.type == _charge_type:
                valid_reaction = True
                # if both clusters are not neutral, only keep reaction if larger cluster is neutral
                if _charge_type == ClusterType.NEUTRAL and not (r1.type == ClusterType.NEUTRAL and r2.type == ClusterType.NEUTRAL):
                    neu_clus = r1 if r1.type == ClusterType.NEUTRAL else r2
                    other_clus = r1 if r1.type != ClusterType.NEUTRAL else r2
                    if neu_clus.total_molecules < other_clus.total_molecules:
                        valid_reaction = False
        else:
            # if charge type is "all", all reactions are valid
            valid_reaction = True
        if valid_reaction:
            # sort to ensure we access the upper triangular outflux matrix
            idx = sorted([r1_idx, r2_idx])
            _outflux = outflux_rates[idx[0], idx[1]]
            reactions.append({
                "cluster_1": r1.label,
                "cluster_2": r2.label,
                "product": results.system.clusters.composition_to_label(collision.product_composition),
                "product_charge": collision.product_charge,
                "flux": _outflux,
            })
            total_flux_out += _outflux

    if math.isclose(total_flux_out, 0,abs_tol=1e-10):
        logger.warning("Total flux out is close to 0. Returning empty list.")
        return [], 0
    threshold_flux = threshold * total_flux_out
    thresholded_reactions = [reaction for reaction in reactions if reaction["flux"] >= threshold_flux]
    thresholded_reactions.sort(key=lambda x: x["flux"], reverse=True)
    return thresholded_reactions, total_flux_out


def plot_final_outflux(
    results: SimulationResults,
    charging_state: str = "all",
    threshold: float = 0.01,
    ax: Optional[Axes] = None,
    output_units: str = "cm^-3 s^-1",
    figsize: Tuple[int, int] = (10, 6),
) -> Axes:
    """Plot the final outflux rates as a pie chart.

    Args:
        results (SimulationResults): Simulation results.
        charging_state (str, optional): Charging state to plot. 
            Defaults to "all". Can be "all", "neutral", "positive", or "negative". 
            At least one of the colliders must have the charge when charging_state is not "all".
        threshold (float, optional): Threshold for significant outflux. Defaults to 0.01.
        ax (Optional[Axes], optional): Axes to plot on. Defaults to None.
        output_units (str, optional): Output units. Defaults to "cm^-3 s^-1".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).

    Returns:
        Axes: Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if charging_state not in ["all", "neutral", "positive", "negative"]:
        raise ValueError(
            f"Invalid charging state: {charging_state}. Must be 'all', 'neutral', 'positive', or 'negative'."
        )

    total_flux = results.get_final_formation_rate(output_units=output_units)
    # format the title
    title = f"Total Flux Out: {total_flux*ureg(output_units):.3g~P}"
    thresholded_reactions, total_flux_charging_state = get_significant_outflux_reactions(results, threshold=threshold, charge_type=charging_state)
    if len(thresholded_reactions) == 0:
        logger.warning(f"No contribution from {charging_state} clusters to the out flux for threshold {threshold}")
        return None
    # convert to output units
    total_flux_charging_state = total_flux_charging_state * ureg("m^-3 s^-1").to(output_units).magnitude
    outfluxes = np.array([reaction["flux"] * ureg("m^-3 s^-1").to(output_units).magnitude for reaction in thresholded_reactions])
    # the amount of other outfluxes in the charging state that are not significant
    other_outflux = total_flux_charging_state - outfluxes.sum()
    outflux_ratios = outfluxes / total_flux_charging_state
    other_outflux_ratio = other_outflux / total_flux_charging_state
    # if Others is less than 1%, label it as "Others: <percentage>%"
    labels = np.array([f"{reaction['cluster_1']} + {reaction['cluster_2']}" for reaction in thresholded_reactions])
    if other_outflux_ratio < 0.01:
        labels = np.concatenate([labels, [f"Others: {other_outflux_ratio * 100:.1f}%"]])
    else:
        labels = np.concatenate([labels, ["Others"]])
    outfluxes = np.concatenate([outflux_ratios, [other_outflux_ratio]])
    # all outfluxes in descending order
    sort_order = np.argsort(outfluxes)[::-1]
    labels = labels[sort_order]
    outfluxes = outfluxes[sort_order]
    if charging_state != "all":
        # how much of the total flux is through the charging state
        percentage_charge_outflux = (total_flux_charging_state / total_flux * 100)
        subtitle = f"Flux out through {charging_state} charge: {total_flux_charging_state*ureg(output_units):.4g~P} ({percentage_charge_outflux:.2f}%)"
        title = f"{title}\n{subtitle}"

    def func(pct):
        if pct < 1:
            return ""
        else:
            return f"{pct:.1f}%"

    wedges, texts, autotexts = ax.pie(
        outfluxes,
        autopct=func,
        startangle=90,
        textprops={"fontsize": 10},
        pctdistance=1.2,
    )
    
    ax.set_title(title)
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.75))
    ax.set_aspect("equal")
    return ax

def reorder_net_fluxes_matrix(
    net_fluxes: np.ndarray, 
    matlab_cluster_labels: list[str],
    python_cluster_labels: list[str]
) -> np.ndarray:
    """
    Reorder the net_fluxes matrix according to the provided mapping.
    
    Parameters
    ----------
    net_fluxes : np.ndarray
        The original net_fluxes matrix (n_clusters x n_clusters)
    matlab_cluster_labels : list[str]
        Labels for the clusters in the Matlab format.
    python_cluster_labels : list[str]
        Labels for the clusters in the Python format.
        
    Returns
    -------
    np.ndarray
        Reordered net_fluxes matrix
    """
    n_clusters = net_fluxes.shape[0]
    mapping = {i: matlab_cluster_labels.index(label) for i,label in enumerate(python_cluster_labels)}
    # Create the reordered matrix
    reordered_matrix = np.zeros_like(net_fluxes)
    
    # Apply the mapping to both rows and columns
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i in mapping and j in mapping:
                new_i = mapping[i]
                new_j = mapping[j]
                reordered_matrix[new_i, new_j] = net_fluxes[i, j]
    
    return reordered_matrix

def plot_net_fluxes_heatmap(
    results: SimulationResults,
    matlab_cluster_labels: list[str]|None = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    output_units: str = "cm^-3 s^-1",
) -> Axes:
    """
    Plot the net_fluxes matrix as a heatmap.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if results.net_fluxes is None:
        raise ValueError("Net fluxes are not available. Please run simulation with `compute_net_fluxes=True`.")
    # find the net fluxes between clusters
    # convert to output units
    net_fluxes = (
        (results.net_fluxes[FluxDirection("cluster", "cluster")] * ureg("m^-3 s^-1"))
        .to(output_units)
        .magnitude
    )
    # reorder the net fluxes matrix if matlab_cluster_labels are provided
    if matlab_cluster_labels is not None:
        reordered_net_fluxes = reorder_net_fluxes_matrix(net_fluxes, matlab_cluster_labels, results.system.clusters.get_labels())
        labels = matlab_cluster_labels  
    else:
        reordered_net_fluxes = net_fluxes
        labels = results.system.clusters.get_labels()
    im = ax.pcolormesh(labels, labels, reordered_net_fluxes, norm="log", edgecolors='k', linewidth=0.5, rasterized=True)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlabel("Target cluster")
    ax.set_ylabel("Source cluster")
    ax.set_aspect("equal")
    fig.colorbar(im,ax=ax,label=f"Net flux ({output_units})")
    return ax

def track_fluxes(
    results: SimulationResults,
    threshold: float = 0.01,
    charge_type: str = "neutral",
) -> tuple[list[dict[str, str|float]], float]:
    """Track the fluxes of a given charge type.

    Args:
        results (SimulationResults): Simulation results.
        threshold (float, optional): Threshold for significant outflux. Defaults to 0.01.
        charge_type (str, optional): Charge type to track. Defaults to "neutral". Can be "neutral", "positive", or "negative".

    Raises:
        ValueError: If invalid charge type is provided.
        ValueError: If both clusters are not neutral when tracking neutral clusters.
        ValueError: If both clusters are not the desired charge when tracking the desired charge type.

    Returns:
        tuple[list[dict[str, str|float]], float]: List of tracked fluxes and the total outflux through the desired charge type.
    """

    if charge_type not in ["neutral", "positive", "negative"]:
        raise ValueError(f"Invalid charge type: {charge_type}. Must be 'neutral', 'positive', or 'negative'.")
    _charge_type = (
        ClusterType.NEUTRAL
        if charge_type == "neutral"
        else ClusterType.POSITIVE
        if charge_type == "positive"
        else ClusterType.NEGATIVE
    )
    
    equations = results.equations
    system = results.system
    clusters = system.clusters
    tracked_fluxes = []
    significant_outflux_reactions, total_flux_out = get_significant_outflux_reactions(results,threshold,charge_type)
    if math.isclose(total_flux_out, 0,abs_tol=1e-10) and len(significant_outflux_reactions) == 0:
        logger.warning(f"No contribution from {charge_type} clusters to the out flux for threshold {threshold}")
        return [], 0, 0
    total_outflux_through_tracked_fluxes = 0
    for reaction in significant_outflux_reactions:
        cluster_1_label = reaction["cluster_1"]
        cluster_2_label = reaction["cluster_2"]
        product_label = reaction["product"]
        cluster_1 = clusters.get_by_label(cluster_1_label)
        cluster_2 = clusters.get_by_label(cluster_2_label)
        # if neutral pathways are desired,
        if _charge_type == ClusterType.NEUTRAL:
            # if both clusters are neutral, track the larger cluster
            if cluster_1.type == ClusterType.NEUTRAL and cluster_2.type == ClusterType.NEUTRAL:
                start = cluster_1.label if cluster_1.total_molecules > cluster_2.total_molecules else cluster_2.label
            elif cluster_1.type == ClusterType.NEUTRAL:
                start = cluster_1.label
            elif cluster_2.type == ClusterType.NEUTRAL:
                start = cluster_2.label
            else:
                raise ValueError(f"Both clusters {cluster_1.label} and {cluster_2.label} are not neutral when tracking {charge_type} clusters.")
        # Otherwise find the cluster with the desired charge and track it
        else:
            if cluster_1.type == _charge_type:
                start = cluster_1.label
            elif cluster_2.type == _charge_type:
                start = cluster_2.label
            else:
                raise ValueError(f"Both clusters {cluster_1.label} and {cluster_2.label} are not {charge_type} when tracking {charge_type} clusters.")
        
        tracked_fluxes.append({
            "start": start,
            "end": product_label,
            "flux": reaction["flux"],
        })
        total_outflux_through_tracked_fluxes += reaction["flux"]

    if len(tracked_fluxes) == 0:
        logger.warning(f"No contribution from {charge_type} clusters to the out flux for threshold {threshold}")
        return []
    

    visited = set()
    to_track = set()
    cluster_labels = clusters.get_labels()
    for flux in tracked_fluxes:
        start_cluster_label = flux["start"]
        
        if start_cluster_label in cluster_labels and start_cluster_label not in visited:
            start_cluster_index = clusters.get_index_by_label(start_cluster_label)
            start_cluster = clusters[start_cluster_index]
            if start_cluster.type == _charge_type:
                logger.debug(f"Tracking flux to {start_cluster_label}")
                origins = find_flux_to_cluster(start_cluster_index,system,equations,results.net_fluxes,same_charge=True)
                total_origin_flux = sum([origin["flux"] for origin in origins])
                for origin in origins:
                    if origin["flux"] / total_origin_flux >= threshold:
                        tracked_fluxes.append({
                            "start": origin["cluster"],
                            "end": start_cluster_label,
                            "flux": origin["flux"],
                        })
                        if origin["cluster"] not in visited:
                            to_track.add(origin["cluster"])
        visited.add(start_cluster_label)
        
    while len(to_track) > 0:
        cluster_label = to_track.pop()
        if cluster_label in visited:
            continue
        if cluster_label not in cluster_labels:
            continue
        cluster_index = clusters.get_index_by_label(cluster_label)
        cluster = clusters[cluster_index]
        if cluster.type != _charge_type:
            continue
        # find the flux from the cluster
        logger.debug(f"Tracking flux to {cluster_label}")
        origins = find_flux_to_cluster(cluster_index,system,equations,results.net_fluxes,same_charge=True)
        total_origin_flux = sum([origin["flux"] for origin in origins])
        for origin in origins:
            if origin["flux"] / total_origin_flux >= threshold:
                tracked_fluxes.append({
                    "start": origin["cluster"],
                    "end": cluster_label,
                    "flux": origin["flux"],
                })
                if origin["cluster"] not in visited:
                    to_track.add(origin["cluster"])
        visited.add(cluster_label)

    tracked_fluxes.sort(key=lambda x: x["flux"], reverse=True)
    
    return tracked_fluxes, total_outflux_through_tracked_fluxes, total_flux_out


def find_flux_to_cluster(
    target_cluster_index: int,
    system: SimulationSystem,
    equations: ClusterEquations,
    net_fluxes: OrderedDict[FluxDirection, np.ndarray],
    same_charge: bool = False,
) -> list[tuple[str, str]]:
    """
    Find the flux from a given cluster.
    """
    target_cluster = system.clusters[target_cluster_index]
    target_cluster_charge = target_cluster.charge
    origins = []
    cluster_fluxes = net_fluxes[FluxDirection("cluster", "cluster")]
    # Collisions that form the cluster
    for collider_i, collider_j in equations.process_tracker.get_collision_birth_terms(
        target_cluster_index
    ):
        collider_i_cluster = system.clusters[collider_i]
        collider_j_cluster = system.clusters[collider_j]
        # # Flux from collisions involving generic ions are not considered
        # if isinstance(collider_i_cluster, GenericIon) or isinstance(collider_j_cluster, GenericIon):
        #     continue
        _flux = cluster_fluxes[collider_i, target_cluster_index]  # should be the same as flux j->target
        # if two identical clusters colliding, the flux needs to be divided by 2,
        #    since it's from the point of view of the colliding parties
        if collider_i == collider_j:
            _flux = _flux / 2
        if _flux > 0:
            small_cluster, large_cluster = system.clusters.sort_by_total_molecules(collider_i_cluster,collider_j_cluster)
            logger.debug(f"Found flux from {large_cluster.label} + {small_cluster.label} -> {target_cluster.label}: {_flux:.3g}")
            if same_charge:
                if large_cluster.charge == target_cluster_charge:
                    origins.append(
                        {
                            "cluster": large_cluster.label,
                            "other": small_cluster.label,
                            "flux": _flux,  
                        }
                    )
                elif small_cluster.charge == target_cluster_charge:
                    origins.append(
                        {
                            "cluster": small_cluster.label, 
                            "other": large_cluster.label,
                            "flux": _flux,
                        }
                    )
                else: # charge of the colliding clusters is different from the target cluster. Take the larger cluster
                    origins.append(
                        {
                            "cluster": large_cluster.label,
                            "other": small_cluster.label,
                            "flux": _flux,
                        }
                    )
            else:
                origins.append(
                    {
                        "cluster": large_cluster.label,
                        "other": small_cluster.label,
                        "flux": _flux,
                    }
                )
    # Evaporations that form the cluster
    for _, evaporator_index in equations.process_tracker.get_evaporation_birth_terms(
        target_cluster_index
    ):
        _flux = cluster_fluxes[evaporator_index, target_cluster_index]
        if _flux > 0:
            origins.append(
                {
                    "cluster": system.clusters[evaporator_index].label,
                    "other": None,
                    "flux": _flux,
                }
            )
    # If there are any boundary collisions that form the cluster, add them to the origins
    if (
        len(
            equations.process_tracker.get_boundary_cluster_birth_terms(
                target_cluster_index
            )
        )
        > 0
    ):
        boundary_fluxes = net_fluxes[FluxDirection("boundary", "cluster")]
        _flux = boundary_fluxes[target_cluster_index]
        if _flux > 0:
            origins.append(
                {
                    "cluster": "boundary",
                    "other": None,
                    "flux": _flux,
                }
            )
    
    # if any boundary collisions that form the cluster, add boundary to the origins
    if (
        len(
            equations.process_tracker.get_boundary_cluster_monomer_birth_terms(
                target_cluster_index
            )
        )
        > 0
    ):
        boundary_monomer_fluxes = net_fluxes[FluxDirection("boundary", "cluster")]
        _flux = boundary_monomer_fluxes[target_cluster_index]
        if _flux > 0:
            origins.append(
                {
                    "cluster": "boundary",
                    "other": None,
                    "flux": _flux,
                }
            )

    # If there are any sources for the cluster, add them to the origins
    source_fluxes = net_fluxes[FluxDirection("source","cluster")]
    _flux = source_fluxes[target_cluster_index]
    if _flux > 0:
        origins.append(
            {
                "cluster": "source",
                "other": None,
                "flux": _flux,
            }
        )

    return origins


def plot_flux_tracking(
    results: SimulationResults,
    threshold: float = 0.01,
    charge_type: str = "neutral",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    n_colors: int = 5,
    output_units: str = "cm^-3 s^-1",
    print_cluster_labels: bool = False,
) -> Axes:
    """
    Plot flux tracking results similar to MATLAB track_fluxes.m
    
    Parameters
    ----------
    tracked_fluxes : list[dict]
        List of flux dictionaries with 'start', 'end', 'flux' keys
    system : System
        The simulation system containing clusters and molecules
    threshold : float
        Threshold for including fluxes in the plot
    charge_type : str
        Type of charge to plot ("neutral", "positive", "negative")
    ax : Optional[Axes]
        Existing axes to plot on. If None, creates new figure and axes.
    figsize : tuple
        Figure size (only used if ax is None)
    n_colors : int
        Number of color levels for flux magnitude
    output_units : str, optional
        Output units for the flux. Default is "cm^-3 s^-1".
    print_cluster_labels : bool
        Whether to print the cluster labels. Default is False.
        
    Returns
    -------
    Axes
        The matplotlib axes object.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyArrowPatch
    

    system = results.system
    # get the tracked fluxes
    tracked_fluxes, total_outflux_through_tracked_fluxes, total_flux_out = track_fluxes(results,threshold=threshold,charge_type=charge_type)
    _charge_type = (
        ClusterType.NEUTRAL
        if charge_type == "neutral"
        else ClusterType.POSITIVE
        if charge_type == "positive"
        else ClusterType.NEGATIVE
    )
    if len(tracked_fluxes) == 0:
        logger.warning(f"No contribution from {charge_type} clusters to the out flux for threshold {threshold}")
        return None
    # convert the tracked fluxes to the desired units
    tracked_fluxes = [
        {
            "start": flux["start"],
            "end": flux["end"],
            "flux": flux["flux"] * ureg("m^-3 s^-1").to(output_units).magnitude,
        }
        for flux in tracked_fluxes
    ]
    # get the total outflux
    total_outflux_through_tracked_fluxes = (total_outflux_through_tracked_fluxes * ureg("m^-3 s^-1")).to(output_units)
    total_flux_out = (total_flux_out * ureg("m^-3 s^-1")).to(output_units)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get neutral molecule names (excluding ions)
    neutral_molecules = system.molecules.get_neutral_molecules()
    molecule_names = list(neutral_molecules.keys())
    # check if the system has 2 neutral molecules
    if len(molecule_names) != 2:
        raise ValueError("This visualization only supports 2-component systems")
    

    # Extract cluster coordinates and flux data
    cluster_coords = {}  # cluster_label -> (x, y) coordinates
    flux_arrows = []  # list of (start_coord, end_coord, flux_value)
    
    # Find all unique clusters and their coordinates
    for flux in tracked_fluxes:
        start_label = flux['start']
        end_label = flux['end']

        # Don't plot fluxes to generic ions
        if end_label == "neg" or end_label == "pos":
            continue
        
        # Get a cluster object from the label
        end_cluster = system.clusters.cluster_from_label(end_label)
        # Find the coordinates of the end cluster by counting the molecules and the corresponding ions
        end_coords = (
            end_cluster.get_molecule_count(molecule_names[0], include_ions=True),
            end_cluster.get_molecule_count(molecule_names[1], include_ions=True)
        )

        # if start is a cluster in the system
        if start_label not in ["neg", "pos"] and start_label in system.clusters:
            # Get a cluster object from the label
            start_cluster = system.clusters.cluster_from_label(start_label)
            start_coords = (
                start_cluster.get_molecule_count(molecule_names[0], include_ions=True),
                start_cluster.get_molecule_count(molecule_names[1], include_ions=True)
            )
            # Draw an arrow if the start and end clusters have the same required charge type
            if start_cluster.type == _charge_type and end_cluster.type == _charge_type:
                cluster_coords[start_label] = start_coords
                cluster_coords[end_label] = end_coords
                flux_arrows.append((start_coords, end_coords, flux['flux']))
            # if the end cluster has the required charge type, but start cluster does not
            elif start_cluster.type != _charge_type and end_cluster.type == _charge_type:
                # Mark the source 
                ax.text(
                    end_coords[0] + 0.5,
                    end_coords[1] + 0.5,
                    start_label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="red",
                    ),
                )
        # if not cluster mark source
        else:
            logger.debug(f"Marking source {start_label} at {end_coords}")
            ax.text(
                end_coords[0] + 0.9,
                end_coords[1] + 0.1,
                start_label,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="black",
                ),
            )
    
    # Find system boundaries (including boundary clusters)
    max_x = max(coord[0] for coord in cluster_coords.values()) if cluster_coords else 0
    max_y = max(coord[1] for coord in cluster_coords.values()) if cluster_coords else 0
    logger.debug(f"Max x: {max_x}, Max y: {max_y}")
    max_x = max(max_x,max_y)
    max_y = max_x
    # Set up the plot
    ax.set_xlabel(f'Number of molecules {molecule_names[0]}')
    ax.set_ylabel(f'Number of molecules {molecule_names[1]}')
    ax.set_title(f'Outgoing flux {total_outflux_through_tracked_fluxes:.4g~P}\nthrough the depicted {charge_type} clusters')
    
    # Set axis limits and grid
    ax.set_xlim(0, max_x + 1)
    ax.set_ylim(0, max_y + 1)
    ax.set_xticks(np.arange(max_x + 1)+0.5,labels=np.arange(max_x + 1))
    ax.set_yticks(np.arange(max_y + 1)+0.5,labels=np.arange(max_y + 1))
    for x in np.arange(max_x + 1):
        ax.axvline(x=x,color='black',linestyle='--',linewidth=0.5)
    for y in np.arange(max_y + 1):
        ax.axhline(y=y,color='black',linestyle='--',linewidth=0.5)
    # ax.grid(True, alpha=0.3)
    
    # Create color map for flux magnitudes
    flux_values = [arrow[2] for arrow in flux_arrows]
    if flux_values:
        min_flux = min(flux_values)
        max_flux = max(flux_values)
        
        # Create log-scaled color levels
        norm = mpl.colors.LogNorm(vmin=min_flux, vmax=max_flux)
        cdict = {
            'red': ((0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),
            'green': ((0.0, 0.45, 0.45),
                    (1.0, 0.0, 0.0)),
            'blue': ((0.0, 0.74, 0.74),
                    (1.0, 0.0, 0.0)),
        }
        cmap = mpl.colors.LinearSegmentedColormap('flux_cmap', cdict, N=n_colors)
        
        
        # Create color map (blue to red)
        # colors = plt.cm.coolwarm(np.linspace(0, 1, n_colors))
        # cmap = LinearSegmentedColormap.from_list('flux_cmap', colors)
        
        # Draw arrows
        for start_coord, end_coord, flux_val in flux_arrows:
            start_coord = (start_coord[0] + 0.5, start_coord[1] + 0.5)
            end_coord = (end_coord[0] + 0.5, end_coord[1] + 0.5)
            # Determine color based on flux magnitude
            # color_idx = np.digitize(flux_val, color_levels) - 1
            # color_idx = max(0, min(color_idx, len(colors) - 1))
            color = cmap(norm(flux_val))
            
            # Determine line width based on flux magnitude
            line_width = 1 + 2 * (flux_val - min_flux) / (max_flux - min_flux)
            
            # Draw arrow
            logger.debug(f"Drawing arrow from {start_coord} to {end_coord} with flux {flux_val}")
            arrow = FancyArrowPatch(
                start_coord, end_coord,
                arrowstyle='-|>',
                mutation_scale=20,
                color=color,
                linewidth=line_width,
                # alpha=0.8
            )
            ax.add_patch(arrow)
    
    # Plot cluster points
    if print_cluster_labels:
        for label, coords in cluster_coords.items():
            # Check if this is a boundary cluster
            is_boundary = label not in system.clusters.get_labels()
            
            if is_boundary:
                # Add cluster labels
                ax.text(
                    coords[0] + 0.5,
                    coords[1] + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="red",
                    ),
                )
            else:
                ax.text(
                    coords[0] + 0.5,
                    coords[1] + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="black",
                    ),
                )
    
    # Add colorbar
    if flux_values:
        norm = mpl.colors.LogNorm(vmin=min_flux, vmax=max_flux)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            ticks=mpl.ticker.LogLocator(base=10),
            format=mpl.ticker.LogFormatterSciNotation(
                base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)
            ),
        )
        cbar.set_label(f'Flux ({output_units})')

    
    # # Draw system boundaries (original system boundary)
    system_max_x = max(cluster.get_molecule_count(molecule_names[0], include_ions=True) 
                      for cluster in system.clusters)
    system_max_y = max(cluster.get_molecule_count(molecule_names[1], include_ions=True) 
                      for cluster in system.clusters)
    
    if system_max_x > 0 and system_max_y > 0:
        ax.hlines(y=system_max_y + 1, xmin=0, xmax=system_max_x + 1, color='black', linestyle='--', linewidth=3)
        ax.vlines(x=system_max_x + 1, ymin=0, ymax=system_max_y + 1, color='black', linestyle='--', linewidth=3)
    
    return ax


def create_flux_graph(system, net_fluxes: OrderedDict[FluxDirection, np.ndarray]) -> nx.DiGraph:
    """
    Create a directed graph from the net_fluxes dictionary.
    
    Parameters
    ----------
    system : System
    net_fluxes : OrderedDict[FluxDirection, np.ndarray]

    Returns
    -------
    graph : nx.DiGraph
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(system.clusters.get_labels())
    special_nodes = {"source","wall","coagulation_sink","boundary","out_neutral","out_positive","out_negative"}
    graph.add_nodes_from(special_nodes)
    # add edges for net fluxes between clusters
    for i in range(system.n_clusters):
        for j in range(system.n_clusters):
            if net_fluxes[FluxDirection("cluster","cluster")][i,j] != 0:
                graph.add_edge(system.clusters.get_label(i),system.clusters.get_label(j),flux=float(net_fluxes[FluxDirection("cluster","cluster")][i,j]))
    net_flux_to_boundary = net_fluxes[FluxDirection("cluster","boundary")] - net_fluxes[FluxDirection("boundary","cluster")]

    for i, flux in enumerate(net_flux_to_boundary.flatten()):
        if flux > 0:
            graph.add_edge(system.clusters.get_label(i),"boundary",flux=float(flux))
        else:
            graph.add_edge("boundary",system.clusters.get_label(i),flux=-float(flux))
   
   # Edges to neutral, positive, and negative outgrowing clusters
    for i, flux in enumerate(net_fluxes[FluxDirection("cluster","out_neutral")].flatten()):
        graph.add_edge(system.clusters.get_label(i),"out_neutral",flux=float(flux))
    for i, flux in enumerate(net_fluxes[FluxDirection("cluster","out_positive")].flatten()):
        graph.add_edge(system.clusters.get_label(i),"out_positive",flux=float(flux))
    for i, flux in enumerate(net_fluxes[FluxDirection("cluster","out_negative")].flatten()):
        graph.add_edge(system.clusters.get_label(i),"out_negative",flux=float(flux))
    
    # Edges to wall, coagulation sink, and dilution losses from clusters
    for i, flux in enumerate(net_fluxes[FluxDirection("cluster","wall")].flatten()):
        graph.add_edge(system.clusters.get_label(i),"wall",flux=float(flux))
    for i, flux in enumerate(net_fluxes[FluxDirection("cluster","coagulation_sink")].flatten()):
        graph.add_edge(system.clusters.get_label(i),"coagulation_sink",flux=float(flux))
    for i, flux in enumerate(net_fluxes[FluxDirection("cluster","dilution")].flatten()):
        graph.add_edge(system.clusters.get_label(i),"dilution",flux=float(flux))
    
    # Edges from source to clusters
    for i, flux in enumerate(net_fluxes[FluxDirection("source","cluster")].flatten()):
        graph.add_edge("source",system.clusters.get_label(i),flux=float(flux))
    return graph


def plot_flux_graph(
    graph: nx.DiGraph,
    results: SimulationResults,
    threshold: float = 0.01,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 10),
    include_sinks: bool = False,
):
    """Plot the flux graph.

    Args:
        graph (nx.DiGraph): The flux graph.
        results (SimulationResults): The simulation results.
        threshold (float, optional): The threshold for the flux. Defaults to 0.01.
        ax (Optional[Axes], optional): The axes to plot on. Defaults to None.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (10, 10).
        include_sinks (bool, optional): Whether to include the wall, coagulation sink, and dilution in the plot. Defaults to False.

    Raises:
        ValueError: If the system is not 2-component.

    Returns:
        Axes: The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    pos = {}
    xscale = 4
    yscale = 10
    total_outflux = results.get_final_formation_rate()
    threshold = threshold * total_outflux

    sink_nodes = ["wall","coagulation_sink","dilution"]
    if not include_sinks:
        induced_edges_data = [(u,v, flux) for u,v,flux in graph.edges.data("flux") if flux >= threshold and u not in sink_nodes and v not in sink_nodes]
    else:
        induced_edges_data = [(u,v, flux) for u,v,flux in graph.edges.data("flux") if flux >= threshold]
    induced_edges = [(u,v) for u,v,flux in induced_edges_data]
    induced_edge_widths = np.array([flux for _,_,flux in induced_edges_data])
    induced_edge_widths = np.log(induced_edge_widths)
    # normalize to 0-1
    induced_edge_widths = (induced_edge_widths - induced_edge_widths.min()) / (induced_edge_widths.max() - induced_edge_widths.min())
    induced_edge_widths = (induced_edge_widths + 0.1) * 5
    induced_graph = graph.edge_subgraph(induced_edges)
    molecule_names = list(results.system.molecules.get_neutral_molecules().keys())
    if len(molecule_names) != 2:    
        raise ValueError("This visualization only supports 2-component systems")
    for cluster in results.system.clusters:
        x_loc = cluster.get_molecule_count(molecule_names[0])
        y_loc = cluster.get_molecule_count(molecule_names[1])
        pos[cluster.label] = (x_loc*xscale,y_loc*yscale)

    max_x = max(i for i,j in pos.values())
    max_y = max(j for i,j in pos.values())

    if include_sinks:
        pos["wall"] = (0,max_y+1*yscale)
        pos["dilution"] = (1*xscale,max_y+2*yscale)
        pos["coagulation_sink"] = (2*xscale,max_y+1*yscale)
    pos["boundary"] = (max_x+1*xscale,max_y/2)
    pos["source"] = (0,0)
    pos["out_neutral"] = (max_x+1*xscale,max_y+1*yscale)
    pos["out_positive"] = (max_x+1*xscale,max_y+2*yscale)
    pos["out_negative"] = (max_x+2*xscale,max_y+1*yscale)
    print(pos)
    node_size = 2000
    nx.draw_networkx_nodes(induced_graph,pos=pos,node_size=node_size,node_shape="o",ax=ax)
    nx.draw_networkx_edges(induced_graph,pos=pos,edge_color="black",width=induced_edge_widths,arrowstyle="-|>",ax=ax,node_size=node_size)
    nx.draw_networkx_labels(induced_graph,pos=pos,font_size=10,ax=ax)
    return ax