"""Dataset loading and processing"""

import logging
import random
import networkx as nx
import numpy as np

from operator import itemgetter

from torch_geometric.utils import from_networkx
from tqdm import tqdm

_LOG = logging.getLogger("gvqvae_logger")

_real_node_pad_val = 0
_pad_node_pad_val = 1
_pad_node_attr_val = 0


def no_features(graph, adj, idx, **kwargs):
    """No features on nodes."""
    return []


def degree_feature(graph, adj, idx, max_degree, one_hot_degree, **kwargs):
    """One-hot degree indicator."""
    node_degree = graph.degree(idx) // 2
    if one_hot_degree:
        feat_vec = np.zeros(max_degree, dtype=int)
        feat_vec[node_degree] = 1
    else:
        feat_vec = np.array([node_degree], dtype=int)
    return feat_vec


def pad_graph(
    nx_graph: nx.DiGraph,
    num_attrs: int,
    make_attrs: bool = True,
    attr_func=None,
    attr_name="attr",
    max_nodes=0,
    max_degree=0,
    one_hot_degree=True,
):
    """Pad a single graph"""
    if make_attrs:
        adj = nx.to_numpy_array(nx_graph)
        nx_graph = nx.from_numpy_array(adj)
        for idx, node in enumerate(nx_graph.nodes):
            nx_graph.nodes[node]["attr"] = attr_func(
                graph=nx_graph,
                adj=adj,
                idx=idx,
                max_nodes=max_nodes,
                max_degree=max_degree,
                one_hot_degree=one_hot_degree,
            )

    nx.set_node_attributes(nx_graph, _real_node_pad_val, "pad")
    pad_nodes = nx.empty_graph(max_nodes, create_using=nx.DiGraph)
    pad_nodes.remove_nodes_from(nx_graph.nodes)
    nx_graph.add_nodes_from(
        pad_nodes,
        attr=[_pad_node_attr_val] * num_attrs,
        pad=_pad_node_pad_val,
    )
    nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))

    return to_pyg(nx_graph, attr_name=attr_name)


def pad_all_graphs(
    nx_graphs,
    num_attrs,
    make_attrs=True,
    attr_func=None,
    attr_name="attr",
    max_nodes=0,
    max_degree=0,
    one_hot_degree=True,
):
    """Pad all graphs in the dataset"""
    graphs = []
    for graph in tqdm(nx_graphs, desc="Converting graphs for pytorch geometric"):
        g = pad_graph(
            graph,
            num_attrs=num_attrs,
            make_attrs=make_attrs,
            attr_func=attr_func,
            attr_name=attr_name,
            max_nodes=max_nodes,
            max_degree=max_degree,
            one_hot_degree=one_hot_degree,
        )
        graphs.append(g)
    return graphs


def to_pyg(padded_nx, attr_name="attr"):
    """Convert one padded graph to PyTorch Geometric format"""
    return from_networkx(padded_nx, group_node_attrs=[attr_name, "pad"])


def pyg_dataset(
    graph_list,
    add_attrs=True,
    existing_attr_name="attr",
    new_attrs="degree",
    one_hot_degree=True,
):
    """Convert list of networkx graphs to PyTorch Geometric format"""
    max_nodes = max([len(g) for g in graph_list])
    if one_hot_degree:
        max_degree = max([max(g.degree(), key=lambda x: x[1])[1] for g in graph_list])
    else:
        max_degree = 1

    if not add_attrs:
        if not all(existing_attr_name in g.nodes[node] for g in graph_list for node in g):
            _LOG.warning(f"Some nodes don't have attribute '{existing_attr_name}'")
        num_attrs = len(graph_list[0].nodes[0].get(existing_attr_name, [0]))
        attr_name = existing_attr_name
        attr_func = None
    else:
        attr_name = "attr"
        attr_func = degree_feature
        num_attrs = max_degree

    pyg_graphs = pad_all_graphs(
        graph_list,
        num_attrs,
        make_attrs=add_attrs,
        attr_func=attr_func,
        attr_name=attr_name,
        max_nodes=max_nodes,
        max_degree=max_degree,
        one_hot_degree=one_hot_degree,
    )
    return pyg_graphs


def load_dataset(
    data,
    proportion_or_count="proportion",
    test_prop=0.2,
    test_num=0,
    add_node_attrs=True,
    attr_type="degree",
    shuffle=True,
    seed=None,
    max_dataset_size=0,
    one_hot_degree=True,
):
    """Load and split the dataset"""
    dataset_size = len(data)
    dataset_indices = list(range(dataset_size))
    if shuffle:
        random.seed(seed)
        random.shuffle(dataset_indices)

    if max_dataset_size > 0:
        dataset_indices = dataset_indices[:max_dataset_size]

    if proportion_or_count == "proportion":
        train_prop = 1 - test_prop
        train_num = int(len(dataset_indices) * train_prop)
    elif proportion_or_count == "count":
        train_num = len(dataset_indices) - test_num
    else:
        raise ValueError(f"Invalid split type '{proportion_or_count}'")

    train_indices = dataset_indices[:train_num]
    test_indices = dataset_indices[train_num:]
    indices = {"train": train_indices, "test": test_indices}

    dataset_getter = itemgetter(*dataset_indices)

    graphs = dataset_getter(data)
    graphs = pyg_dataset(
        graph_list=graphs,
        add_attrs=add_node_attrs,
        new_attrs=attr_type,
        one_hot_degree=one_hot_degree,
    )

    train_graphs = graphs[:train_num]
    test_graphs = graphs[train_num:]

    return train_graphs, test_graphs, indices
