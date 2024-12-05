"""Functions to run graph reconstructions"""

import logging
import os
import pickle as pkl
import time

import torch

from pathlib import Path

from tqdm import tqdm

from vqt2g.run_vqt2g import setup_gvqvae_model, VQT2GConfig
from vqt2g.utils.gvqvae_utils import load_gvqvae
from vqt2g.utils.recon_utils import heatmap
from vqt2g.generate import reconstruct_graph, plot_real_and_generated_graph
_LOG = logging.getLogger("gvqvae_logger")

def recon_runner(
    config: VQT2GConfig,
    num_recon: int = 30,
    threshold: float = 0.5,
    edge_sampling: bool = False,
    heatmap_padding: bool = False,
    comment=None,
):
    """Run graph reconstructions using the GVQVAE model"""
    cfg_base = config.config
    device = torch.device(cfg_base.device)

    test_dir = Path(cfg_base.this_run_dir, "recons")
    os.makedirs(test_dir, exist_ok=True)

    # Load graphs
    data_dir = config.dataset.dataset_save_path
    with open(Path(data_dir, "train_graphs.pkl"), "rb") as f:
        train_graphs = pkl.load(f)
    with open(Path(data_dir, "test_graphs.pkl"), "rb") as f:
        test_graphs = pkl.load(f)

    num_recon = min(num_recon, len(train_graphs), len(test_graphs))

    # Load GVQVAE
    gvqvae_model = setup_gvqvae_model(config, device)
    load_gvqvae(config, gvqvae_model, device)
    _LOG.info(f"Loaded GVQVAE checkpoint from {config.gvqvae.test.last_checkpoint}")

    # Make folders
    eval_run_name = f"recons_{time.strftime('%b_%d_%H-%M')}"
    this_eval_dir = Path(test_dir, eval_run_name)
    os.makedirs(this_eval_dir, exist_ok=True)

    recons_dir = Path(this_eval_dir, "recons")
    heats_dir = Path(this_eval_dir, "heatmaps")
    os.makedirs(recons_dir, exist_ok=True)
    os.makedirs(heats_dir, exist_ok=True)

    if comment is not None:
        comment_file = Path(this_eval_dir, "recon_comment.txt")
        with open(comment_file, "w") as f:
            f.write(comment)

    _LOG.info(f"Reconstructing {num_recon} graphs from train and test sets")
    for idx in tqdm(range(num_recon)):
        train_graph = train_graphs[idx]
        test_graph = test_graphs[idx]

        train_rec = reconstruct_graph(
            gvqvae_model, train_graph, thresh=threshold, edge_sampling=edge_sampling, device=device
        )
        test_rec = reconstruct_graph(
            gvqvae_model, test_graph, thresh=threshold, edge_sampling=edge_sampling, device=device
        )

        # Plot original graph against reconstructed graph
        train_fname = f"train_recon_{idx:03d}.png"
        test_fname = f"test_recon_{idx:03d}.png"
        plot_real_and_generated_graph(
            real_graph=train_graph,
            generated_graph=train_rec,
            text=f"Train recon {idx:03d}",
            save_folder=recons_dir,
            file_name=train_fname,
        )

        plot_real_and_generated_graph(
            real_graph=test_graph,
            generated_graph=test_rec,
            text=f"Test recon {idx:03d}",
            save_folder=recons_dir,
            file_name=test_fname,
        )

        # Generate heatmaps
        train_heat_fname = f"train_heatmap_{idx:03d}.png"
        test_heat_fname = f"test_heatmap_{idx:03d}.png"
        heatmap(
            gvqvae_model,
            train_graph,
            title_text=f"Train graph {idx:03d}",
            include_padding=heatmap_padding,
            full_range=True,
            device=device,
            save_folder=heats_dir,
            file_name=train_heat_fname,
        )
        heatmap(
            gvqvae_model,
            test_graph,
            title_text=f"Test graph {idx:03d}",
            include_padding=heatmap_padding,
            full_range=True,
            device=device,
            save_folder=heats_dir,
            file_name=test_heat_fname,
        )
    _LOG.info("Done")
