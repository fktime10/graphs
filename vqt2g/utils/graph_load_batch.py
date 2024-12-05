import os
import networkx as nx
import numpy as np

def graph_load_batch(
    data_dir,
    min_num_nodes=100,
    max_num_nodes=500,
    name="DD",
):
    """Load graphs from a dataset.

    Args:
        data_dir (str): Directory containing the dataset.
        min_num_nodes (int): Minimum number of nodes per graph.
        max_num_nodes (int): Maximum number of nodes per graph.
        name (str): Name of the dataset.

    Returns:
        list: A list of networkx graphs.
    """
    G = nx.Graph()
    path = os.path.join(data_dir, name)
    data_adj = np.loadtxt(os.path.join(path, "{}_A.txt".format(name)), delimiter=",").astype(int)
    data_node_label = np.loadtxt(
        os.path.join(path, "{}_node_labels.txt".format(name)), delimiter=","
    ).astype(int)
    data_graph_indicator = np.loadtxt(
        os.path.join(path, "{}_graph_indicator.txt".format(name)), delimiter=","
    ).astype(int)

    data_tuple = list(map(tuple, data_adj))

    # Add edges
    G.add_edges_from(data_tuple)
    for i in range(data_node_label.shape[0]):
        G.add_node(i + 1, label=data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # Find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        if min_num_nodes <= G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
    return graphs
