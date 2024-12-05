"""Compute the MMD stats from two sets of pickled graphs"""

import logging

import networkx as nx
import numpy as np

from collections import defaultdict

from vqt2g.stats.disc_fns import (
    degree_stats,
    clustering_stats,
    spectral_stats,
    orbit_stats_all,
    compute_mmd,
)


_LOG = logging.getLogger("vqt2g_logger")


def _fmt(stat):
    """Format stats nicely for the debug logger"""
    return f"{stat:.4e}" if stat < 0.01 else f"{stat:.4f}"


def graph_stats_mmd(graphs_1, graphs_2, kernel, sigmas):
    """Compute MMD stats between `graphs_1` and `graphs_2`"""

    #    if isinstance(sigmas, list):
    #        sig1, sig2, sig3, sig4 = sigmas
    #    else:
    #        sig1 = sig2 = sig3 = sig4 = sigmas

    # These are hardcoded for consistency with baseline graph generation models
    sig1 = 1.0
    sig2 = 30.0
    sig3 = 0.1
    sig4 = 1.0

    mmd_degree = degree_stats(graphs_1, graphs_2, kernel, sigma=sig1)
    _LOG.debug(f"MMD degree with kernel {kernel}, sigma {sig1} = {_fmt(mmd_degree)}")
    mmd_4orbits = orbit_stats_all(graphs_1, graphs_2, kernel, sigma=sig2)
    _LOG.debug(f"MMD 4orbits with kernel {kernel}, sigma {sig2} = {_fmt(mmd_4orbits)}")
    mmd_clustering = clustering_stats(graphs_1, graphs_2, kernel, sigma=sig3)
    _LOG.debug(f"MMD clustering with kernel {kernel}, sigma {sig3} = {_fmt(mmd_clustering)}")
    mmd_spectral = spectral_stats(graphs_1, graphs_2, kernel, sigma=sig4)
    _LOG.debug(f"MMD spectral with kernel {kernel}, sigma {sig4} = {_fmt(mmd_spectral)}")

    num_nodes_1 = [len(g) for g in graphs_1]
    num_nodes_2 = [len(g) for g in graphs_2]
    mmd_num_nodes = compute_mmd(
        [np.bincount(num_nodes_1)],
        [np.bincount(num_nodes_2)],
        kernel=kernel,
    )

    return mmd_num_nodes, mmd_degree, mmd_clustering, mmd_spectral, mmd_4orbits


def compute_stats(graphs1, graphs2, kernel, sigma, comment="", remove_isolated=True):
    """Compute metrics between two sets of graphs"""

    # If a text-graph dataset loaded, pick the graphs only
    if isinstance(graphs1[0], tuple):
        graphs1 = [i[0] for i in graphs1]
    if isinstance(graphs2[0], tuple):
        graphs2 = [i[0] for i in graphs2]

    # Ensure they're undirected. Later parts can sometimes break if they're not
    graphs1 = [nx.to_undirected(g) for g in graphs1]
    graphs2 = [nx.to_undirected(g) for g in graphs2]

    # Remove isolated nodes from both sets of graphs - they're useless
    # This also keeps consisistent with baseline models
    first_set, second_set = [], []
    for g in graphs1:
        if remove_isolated and nx.number_connected_components(g) > 1:
            g = g.subgraph(max(nx.connected_components(g), key=len))
        first_set.append(g)
    for g in graphs2:
        if remove_isolated and nx.number_connected_components(g) > 1:
            g = g.subgraph(max(nx.connected_components(g), key=len))
        second_set.append(g)

    # Compute all 4 stats
    mmd_nodes, mmd_deg, mmd_clust, mmd_spec, mmd_orbits = graph_stats_mmd(
        first_set, second_set, kernel, sigma
    )

    # Log the stats then return them for the runner to save
    if comment != "":
        comment += "\n"
    _LOG.info(
        f"{comment}MMD scores:\nNum nodes  = {mmd_nodes:07f}\nDegree     = {mmd_deg:07f}\n"
        f"Clustering = {mmd_clust:07f}\nSpectral   = {mmd_spec:07f}\n4-orbits   = {mmd_orbits:07f}"
    )

    stats_dict = {
        "mmd_num_nodes": mmd_nodes,
        "mmd_degree": mmd_deg,
        "mmd_clustering": mmd_clust,
        "mmd_spectral": mmd_spec,
        "mmd_4_orbits": mmd_orbits,
    }
    if comment != "":
        stats_dict["comment"] = comment
    return stats_dict


def egos_topic_stats(
    graphs_test, graphs_gen, texts, topic_map, kernel, sigma, remove_isolated=True
):
    """Similar to above, but calculates the stats by in-topic and out-topic separately. Used for
    the wiki ego dataset
    """

    topics = sorted(list(topic_map.keys()))

    # Entire test set need not be generated
    num_graphs = len(graphs_gen)

    # Invert the topic dict
    article_topic = {}
    for key, vals in topic_map.items():
        for v in vals:
            article_topic[v] = key

    # Get the title part of each text
    titles = []
    for t in texts:
        txt = t.replace("<TITLE>", "")
        txt = txt.split("<SENTENCE>")[0]
        titles.append(txt)

    # Match up each test and generated graph with its topic
    test_dict = defaultdict(list)
    gen_dict = defaultdict(list)

    for idx in range(num_graphs):

        title = titles[idx]
        topic = article_topic[title]

        # Keys are topic name
        # Values are the list of graphs from that topic
        test_dict[topic].append(graphs_test[idx])
        gen_dict[topic].append(graphs_gen[idx])

    # Compute within-topic stats for each topic
    within_topic_stats = []

    for idx, topic in enumerate(topics):

        _LOG.info(f"Starting within-topic stats for {topic}")

        topic_test_graphs = test_dict[topic]
        topic_gen_graphs = gen_dict[topic]

        # If either dict has none of that topic
        if len(topic_gen_graphs) <= 1 or len(topic_test_graphs) == 0:

            _LOG.warning(f"Skipping in-stats for {topic}, a list is empty")
            # Nested 'if' so can know if both lists empty
            if len(topic_gen_graphs) <= 1:
                _LOG.warning("Generated graphs empty for {topic}")
            if len(topic_test_graphs) == 0:
                _LOG.warning("Test graphs empty for {topic}")
            continue
        _LOG.info(
            f"{topic} within-topic stats: test set has {len(topic_test_graphs)} graphs, "
            f"generated set has {len(topic_gen_graphs)} graphs"
        )

        in_stats = compute_stats(
            graphs1=topic_test_graphs,
            graphs2=topic_gen_graphs,
            kernel=kernel,
            sigma=sigma,
            comment=topic,
            remove_isolated=remove_isolated,
        )
        within_topic_stats.append(in_stats)

    _LOG.info(f"Finished all within-topic stats")

    # Compute means
    in_degs = float(np.mean([i["mmd_degree"] for i in within_topic_stats]))
    in_clus = float(np.mean([i["mmd_clustering"] for i in within_topic_stats]))
    in_orbs = float(np.mean([i["mmd_4_orbits"] for i in within_topic_stats]))
    in_spec = float(np.mean([i["mmd_spectral"] for i in within_topic_stats]))
    _LOG.info(f"MMD within-topic degree:       {_fmt(in_degs)}")
    _LOG.info(f"MMD within-topic clustering:   {_fmt(in_clus)}")
    _LOG.info(f"MMD within-topic 4orbits:      {_fmt(in_orbs)}")
    _LOG.info(f"MMD within-topic spectral:     {_fmt(in_spec)}")

    # Out-of-topic stats
    out_topic_stats = []
    for idx, topic in enumerate(topics):
        _LOG.info(f"Starting out-topic stats for {topic}")

        topic_gen_graphs = gen_dict[topic]
        # Get all non-topic datasets
        out_topic_test_graphs = []
        for t in topics:  # 'topic' used in parent loop, using 't' instead
            if t == topic:
                continue
            out_topic_test_graphs += test_dict[topic]

        if len(topic_gen_graphs) <= 1 or len(out_topic_test_graphs) == 0:
            _LOG.warning(f"Skipping out-stats for {topic}")
            continue

        _LOG.info(
            f"For {topic} out-topic stats: non-topic test set has {len(out_topic_test_graphs)} "
            f"graphs, generated set has {len(topic_gen_graphs)} graphs"
        )

        out_stats = compute_stats(
            graphs1=out_topic_test_graphs,
            graphs2=topic_gen_graphs,
            kernel=kernel,
            sigma=sigma,
            comment=topic,
            remove_isolated=remove_isolated,
        )

        out_topic_stats.append(out_stats)

    _LOG.info("Finished all out-stats")
    out_degs = float(np.mean([i["mmd_degree"] for i in out_topic_stats]))
    out_clus = float(np.mean([i["mmd_clustering"] for i in out_topic_stats]))
    out_orbs = float(np.mean([i["mmd_4_orbits"] for i in out_topic_stats]))
    out_spec = float(np.mean([i["mmd_spectral"] for i in out_topic_stats]))
    _LOG.info(f"MMD out-topic degree:       {_fmt(out_degs)}")
    _LOG.info(f"MMD out-topic clustering:   {_fmt(out_clus)}")
    _LOG.info(f"MMD out-topic orbsits:      {_fmt(out_orbs)}")
    _LOG.info(f"MMD out-topic spectral:     {_fmt(out_spec)}")

    return [within_topic_stats, out_topic_stats]
