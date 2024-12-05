import logging
import random

import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch_geometric as pyg
import numpy as np
import networkx as nx
from pathlib import Path

from torch_geometric.data import Data as pyg_Data

from vqt2g.gvqvae import GVQVAE
from vqt2g.utils.gvqvae_utils import real_num_nodes

_LOG = logging.getLogger("gvqvae_logger")


def reconstruct_graph(
    model: GVQVAE,
    graph: pyg_Data,
    thresh: float = 0.8,
    edge_sampling: bool = False,
    device=torch.device("cuda:0"),
):
    num_nodes = real_num_nodes(model, graph=graph)
    num_edges = graph.edge_index.size(1) // 2
    edge_probs = model.encode_decode_graph(graph, num_nodes=num_nodes, device=device)
    edge_topk = not edge_sampling
    recon_graph = edge_probs_to_graph(
        model=model,
        edge_probs=edge_probs,
        num_nodes=num_nodes,
        num_edges=num_edges,
        thresh=thresh,
        edge_topk=edge_topk,
    )
    return recon_graph

def edge_probs_to_graph(
    model: GVQVAE,
    edge_probs: torch.Tensor,
    num_nodes: int,
    num_edges: int = 0,
    thresh: float = 0.9,
    edge_topk: bool = True,
    extra_edge_randomness: bool = False,
) -> nx.DiGraph:
    """Convert edge probabilities to a directed graph."""
    pairs = model._all_edges(num_nodes)
    pairs = pairs.cpu().numpy()

    graph_edges = []

    # Stage 1: For each node, select the most probable outgoing edge
    for n in range(num_nodes):
        # Collect all edges starting from node n
        edges_from_node = [(edge_idx, edge) for edge_idx, edge in enumerate(pairs.T) if edge[0] == n]
        if not edges_from_node:
            continue
        # Get probabilities for these edges
        probs_from_node = [(edge_probs[edge_idx].item(), edge_idx) for edge_idx, _ in edges_from_node]
        # Find the edge with the highest probability
        highest_prob, highest_edge_idx = max(probs_from_node, key=lambda x: x[0])
        # Add the edge if the probability exceeds the threshold
        if highest_prob >= thresh:
            edge = pairs[:, highest_edge_idx]
            graph_edges.append(tuple(edge))

    # Stage 2: Add remaining edges based on threshold
    for idx, prob in enumerate(edge_probs):
        if prob >= thresh and idx not in [e[1] for e in graph_edges]:
            edge = pairs[:, idx]
            graph_edges.append(tuple(edge))

    # If `num_edges` is specified, limit the number of edges
    if num_edges > 0 and len(graph_edges) > num_edges:
        # Sort edges by probability
        edge_probs_list = [(idx, prob.item()) for idx, prob in enumerate(edge_probs)]
        edge_probs_list.sort(key=lambda x: x[1], reverse=True)
        top_edge_indices = [idx for idx, _ in edge_probs_list[:num_edges]]
        graph_edges = [tuple(pairs[:, idx]) for idx in top_edge_indices]

    # Create the directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(graph_edges)
    return G

def select_num_nodes(edge_probs, max_nodes):
    # Placeholder implementation
    return max_nodes

def plot_real_and_generated_graph(
    real_graph,
    generated_graph,
    save_folder,
    file_name,
    text=None,
    text_as_title=True,
    plot_width=20,
    plot_height=10,
    graphviz_layout=False,
    graphviz_prog="neato",
    node_id_labels=False,
):
    """Plot a real graph and a generated graph side by side."""
    plt.figure(figsize=(plot_width, plot_height))
    plt.clf()
    if text_as_title and text is not None:
        plt.suptitle(text)

    # Plot Real Graph
    plt.subplot(121)
    real_nxg = pyg.utils.to_networkx(real_graph)
    real_nxg = nx.DiGraph(real_nxg)  # Use nx.Graph() if graphs are undirected
    real_nxg.remove_nodes_from(list(nx.isolates(real_nxg)))  # Remove isolated nodes
    plt.title(f"Real graph - {len(real_nxg)} nodes, {real_nxg.number_of_edges()} edges")
    if graphviz_layout:
        pos = nx.nx_pydot.graphviz_layout(real_nxg, prog=graphviz_prog)
    else:
        pos = nx.spring_layout(real_nxg)
    nx.draw(
        real_nxg,
        pos=pos,
        with_labels=node_id_labels,
        node_size=150,
        width=0.7,
    )
    # Ensure generated_graph is a NetworkX graph
    if not isinstance(generated_graph, nx.Graph):
        raise TypeError(f"generated_graph is not a NetworkX graph. It is {type(generated_graph)}")

    plt.subplot(122)
    plt.title(f"Generated graph - {len(generated_graph)} nodes, {generated_graph.number_of_edges()} edges")
    if graphviz_layout:
        pos = nx.nx_pydot.graphviz_layout(generated_graph, prog=graphviz_prog)
    else:
        pos = nx.spring_layout(generated_graph)
    nx.draw(
        generated_graph,
        pos=pos,
        with_labels=node_id_labels,
        node_size=150,
        width=0.7,
    )
    # Save the plot
    save_path = Path(save_folder) / file_name
    plt.savefig(save_path)
    plt.close()