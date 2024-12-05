"""MMD statistics functions"""

### Adapted from GRAN

import os
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures

# import pyemd

from pathlib import Path

from scipy.linalg import eigvalsh  # , toeplitz
from functools import partial

from vqt2g.stats.kernels import gaussian, gaussian_emd, gaussian_tv


_kernel_map = {
    "gaussian": gaussian,
    "gaussian_tv": gaussian_tv,
    "gaussian_emd": gaussian_emd,
}

# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
COUNT_START_STR = "orbit counts:"


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def discrepancy(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """Discrepancy between 2 samples"""

    kernel = _kernel_map[kernel]

    d = 0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dist in executor.map(
                kernel_parallel_worker,
                [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1],
            ):
                d += dist
    d /= len(samples1) * len(samples2)
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """MMD between two samples"""

    if is_hist:  # normalize histograms into pmf
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

    disc_1_1 = discrepancy(samples1, samples1, kernel, *args, **kwargs)
    disc_2_2 = discrepancy(samples2, samples2, kernel, *args, **kwargs)
    disc_1_2 = discrepancy(samples1, samples2, kernel, *args, **kwargs)
    return disc_1_1 + disc_2_2 - 2 * disc_1_2


def compute_emd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """EMD between average of two samples"""

    if is_hist:  # normalize histograms into pmf
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    return discrepancy(samples1, samples2, kernel, *args, **kwargs), [samples1[0], samples2[0]]


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, kernel, sigma, is_parallel=True):
    """Compute the distance between the degree distributions of two unordered sets of graphs.

    Args:
      graph_ref_list: list of networkx graphs to be evaluated
      graph_pred_list: list of networkx graphs to be evaluated
      kernel: kernel for the distance computation (e.g. gaussian_tv)
      sigma: 
      is_parallel:

    Returns:

    """

    sample_ref, sample_pred = [], []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if len(G) > 0]
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel, sigma=sigma)
    return mmd_dist


def spectral_worker(G):
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, kernel, sigma, is_parallel=True):
    """Compute distance between the degree distributions of two unordered sets of graphs.
    graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated

    Args:
      graph_ref_list:
      graph_pred_list:
      kernel:
      sigma:
      is_parallel:  (Default value = True)

    Returns:

    """

    sample_ref, sample_pred = [], []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if len(G) > 0]
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=kernel, sigma=sigma)
    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clust_coefs = list(nx.clustering(G).values())
    hist, _ = np.histogram(clust_coefs, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, kernel, sigma, bins=100, is_parallel=True):
    """

    Args:
      graph_ref_list:
      graph_pred_list:
      kernel:
      sigma:
      bins:  (Default value = 100)
      is_parallel:  (Default value = True)

    Returns:

    """

    sample_ref, sample_pred = [], []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if len(G) > 0]
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)
        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)
    return compute_mmd(sample_ref, sample_pred, kernel=kernel, sigma=sigma)


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1
    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):

    # If orca isn't executable then it'll fail
    orca_path = Path("vqt2g", "stats", "utils", "orca", "orca")
    can_execute = os.access(str(orca_path), os.X_OK)
    if not can_execute:
        try:
            os.chmod(orca_path, 0o777)
        except PermissionError as e:
            raise PermissionError("User can't execute orca, orbit stats will fail") from e
        # Double check
        if not os.access(str(orca_path), os.X_OK):
            raise PermissionError("Orca isn't executable, permissions must be manually fixed")

    tmp_fname = "vqt2g/stats/utils/orca/tmp.txt"
    f = open(tmp_fname, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()

    output = sp.check_output(
        ["./vqt2g/stats/utils/orca/orca", "node", "4", "vqt2g/stats/utils/orca/tmp.txt", "std"]
    )
    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list, kernel, sigma):
    """

    Args:
      graph_ref_list:
      graph_pred_list:
      kernel:
      sigma:

    Returns:

    """

    total_counts_ref = []
    total_counts_pred = []

    for G in graph_ref_list:
        G = nx.from_numpy_array(nx.to_numpy_array(G))
        try:
            orbit_counts = orca(G)
        except PermissionError as e:
            raise PermissionError("Orca permission error in ref graphs") from e
            # continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except PermissionError as e:
            raise PermissionError("Orca permission error in ref graphs") from e
            # continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    mmd_dist = compute_mmd(
        total_counts_ref,
        total_counts_pred,
        kernel=kernel,
        is_hist=False,
        sigma=sigma,
    )

    return mmd_dist
