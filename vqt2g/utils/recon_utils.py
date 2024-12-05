"""Functions for graph reconstruction"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch

from pathlib import Path
import torch_geometric.utils as pyg_utils
from vqt2g.utils.gvqvae_utils import real_num_nodes


def reshape_to_adj(adj_vector, size, diag_val=1.0):
    """Reshape GVQVAE output into a symmetric matrix"""
    adj_dims = (size, size)
    mask = torch.zeros(adj_dims)
    tril_inds = torch.tril_indices(size, size, offset=-1)
    mask[tril_inds[0], tril_inds[1]] = adj_vector
    adj = np.diag([diag_val] * size)
    return mask + mask.T + adj

import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.utils as pyg_utils

def heatmap(
    model,
    graph,
    title_text="Graph Heatmap",
    include_padding=False,
    full_range=True,
    device=torch.device("cuda:0"),
    save_folder=None,
    file_name=None,
):
    plt.figure(figsize=(20, 10))
    plt.clf()

    plt.subplot(121)
    # Convert edge_index to edge list
    edge_index = graph.edge_index.cpu().numpy()
    edge_list = list(zip(edge_index[0], edge_index[1]))
    # Create NetworkX graph
    real_nxg = nx.Graph()
    real_nxg.add_edges_from(edge_list)
    real_nxg.add_nodes_from(range(graph.num_nodes))  # Ensure all nodes are added
    real_nxg.remove_nodes_from(list(nx.isolates(real_nxg)))  # Remove isolated nodes if needed
    plt.title(f"{title_text} - {len(real_nxg)} nodes, {real_nxg.number_of_edges()} edges")
    pos = nx.spring_layout(real_nxg)
    nx.draw(
        real_nxg,
        pos=pos,
        with_labels=False,
        node_size=150,
        width=0.7,
    )

    # Heatmap plot (You may need to adjust this part based on your implementation)
    plt.subplot(122)
    num_nodes = graph.num_nodes
    # Generate edge probabilities using your model
    with torch.no_grad():
        edge_probs = model.encode_decode_graph(graph, num_nodes=num_nodes, device=device)
    # Reshape edge_probs into a matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    pairs = model._all_edges(num_nodes)
    for idx, (src, dst) in enumerate(pairs.cpu().t()):
        adj_matrix[src, dst] = edge_probs[idx].cpu()

    # Convert adjacency matrix to numpy array for plotting
    adj_matrix_np = adj_matrix.cpu().numpy()
    plt.title("Edge Probability Heatmap")
    plt.imshow(adj_matrix_np, cmap='hot', interpolation='nearest')
    plt.colorbar()

    if save_folder and file_name:
        save_path = Path(save_folder) / file_name
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()