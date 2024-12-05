"""Utilities for GVQVAE training"""

import logging
import os
import os.path as osp
from pathlib import Path
from datetime import datetime as dt
import random
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

_LOG = logging.getLogger("gvqvae_logger")


def dt_fmt(timer):
    """Format time as HH:MM:SS"""
    return str(timer).split(".")[0]


def real_num_nodes(model, graph):
    """Get the number of nodes in the graph, ignoring padding nodes"""
    return model.decoder.max_nodes - int(graph.x[:, -1].sum().item())


@torch.no_grad()
def check_unused_codes(
    model,
    dataset,
    device,
    max_codebook_samples=2000,
):
    """Encode items in dataset, return code usage stats.

    Args:
        model: GVQVAE model.
        dataset: Training dataset.
        device: CPU/CUDA device to use.
        max_codebook_samples (int): Max items from dataset to use.

    Returns:
        tuple: Number of dead codes, most used code, its count, total codes, and number of almost dead codes.
    """
    almost_dead_thresh = 10
    codes_list = []

    codebook_size = model.codebook_size
    if len(dataset) > max_codebook_samples:
        dataset = random.sample(dataset, max_codebook_samples)
    model.eval()
    # Encode every graph, get list of codes
    for graph in dataset:
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        codes = model.vq_encode(x=x, edge_index=edge_index, only_return_ids=True)
        codes = codes.view(-1).cpu().tolist()
        codes_list += codes

    total_codes = len(codes_list)
    codes_used = set(codes_list)  # Set of codes used at least once
    dead_codes = set(range(codebook_size)) - codes_used  # Completely dead codes
    codes_counter = Counter(codes_list)

    top_code, top_code_count = codes_counter.most_common(1)[0]
    almost_dead_codes = len([1 for _, v in codes_counter.items() if v <= almost_dead_thresh])

    return len(dead_codes), top_code, top_code_count, total_codes, almost_dead_codes


def codebook_check(
    model,
    codebook_dataset,
    device,
    step_num,
    max_codebook_samples=0,
):
    """Check codebook usage stats like the number of dead codes"""
    codecheck_start = dt.now()

    num_dead, top_code, top_code_count, total_codes, almost_dead = check_unused_codes(
        model=model,
        dataset=codebook_dataset,
        device=device,
        max_codebook_samples=max_codebook_samples,
    )
    top_code_freq = top_code_count / total_codes

    codecheck_time = str(dt.now() - codecheck_start)[5:-4]
    fp = (
        f"Codebook usage at {step_num:4d}: {num_dead:3d}/{model.codebook_size} dead codes, "
        f"{almost_dead:2d} nearly dead, most used code: {top_code:3d}, "
        f"{100 * top_code_freq:.02f}%  ({top_code_count}/{total_codes}"
    )
    if step_num == 1:
        fp += f", perfect={100/model.codebook_size:.02f}%"
    _LOG.info(f"{fp}), took {codecheck_time}s\n")


@torch.no_grad()
def codebook_refit(
    model,
    dataset,
    device,
    kmeans_dataset_size,
    step_num,
):
    """Refit codebook using k-means clustering"""
    refit_start = dt.now()

    num_graphs = int(np.ceil(kmeans_dataset_size / model.max_nodes))
    if num_graphs < len(dataset):
        dataset = random.sample(dataset, num_graphs)
    model.eval()
    # Encode every graph, get list of vectors
    node_emb_list = []
    for graph in dataset:
        node_embs = model.encode_graph(graph, to_codebook_vectors=False, device=device)
        node_embs = node_embs.tolist()
        node_emb_list += node_embs
    model.vq.refit_codebook_kmeans(np.array(node_emb_list), batch_size=4096, max_iter=1000)
    refit_time = str(dt.now() - refit_start)[5:-4]
    _LOG.info(f"Refit codebook after step {step_num}, took {refit_time}")


def save_gvqvae(config, save_folder, fname, model, step_num, optimizer=None, scheduler=None):
    """Save the model weights and the optimizer/scheduler state"""
    checkpoint = {"model": model.state_dict(), "step": step_num}
    ckpt_path = Path(save_folder, fname)
    torch.save(checkpoint, ckpt_path)

    # Save optimizer + scheduler, overwrite previous if it exists
    opt_ckpt = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
    opt_path = Path(save_folder, "optimizer.pt")
    torch.save(opt_ckpt, opt_path)

    config.gvqvae.test.last_checkpoint = fname
    config.save_config("gvqvae")


def load_gvqvae(config, model, device, optimizer=None, scheduler=None):
    """Load saved model for inference or resuming training"""
    ckpt_dir = config.gvqvae.checkpoint_dir
    ckpt_name = config.gvqvae.test.last_checkpoint
    ckpt_path = Path(ckpt_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    model.to(device)

    if optimizer is not None:
        opt_path = Path(ckpt_dir, "optimizer.pt")
        opt = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(opt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(opt["scheduler"])


def plot_losses(
    total_losses, recon_errors, vq_losses, perplexities, run_name, step_num, run_folder
):
    """Plot the total loss, reconstruction loss, perplexity, and VQ loss from training"""
    fname = f"{run_name}_losses_{step_num:05d}"
    title = f"Losses at step {step_num}"

    lpfolder = os.path.join(run_folder, "loss_plots")
    if not os.path.exists(lpfolder):
        os.mkdir(lpfolder)

    # Set different moving average window depending on step number
    N = max(100, step_num // 100)
    off = max(0, step_num // 10)
    end = 1000000

    plt.figure(figsize=(15, 15))
    plt.clf()

    plt.subplot(411)
    moving_avg = uniform_filter1d(total_losses, size=N)
    plt.title(
        f"Autoencoder training loss, {N}-step MA from step {off}-{step_num}. Min MA loss val: {round(min(moving_avg),4)}"
    )
    plt.grid(axis="y")
    plt.plot(range(off + 1, min(end + 1, len(total_losses) + 1)), moving_avg[off:end])

    plt.subplot(412)
    moving_avg = uniform_filter1d(recon_errors, size=N)
    plt.title(
        f"Autoencoder training recon loss, {N}-step MA from step {off}-{step_num}. Min MA recon loss val: {round(min(moving_avg),4)}"
    )
    plt.grid(axis="y")
    plt.plot(range(off + 1, min(end + 1, len(recon_errors) + 1)), moving_avg[off:end])

    plt.subplot(413)
    moving_avg = uniform_filter1d(vq_losses, size=N)
    plt.title(
        f"Autoencoder training VQ loss, {N}-step MA from step {off}-{step_num}. Min MA VQ loss val: {round(min(moving_avg),4)}"
    )
    plt.grid(axis="y")
    plt.plot(range(off + 1, min(end + 1, len(vq_losses) + 1)), moving_avg[off:end])

    plt.subplot(414)
    moving_avg = uniform_filter1d(perplexities, size=N)
    plt.title(
        f"Autoencoder training perplexity, {N}-step MA from step {off}-{step_num}. Max MA perplexity val: {round(max(moving_avg),2)}"
    )
    plt.grid(axis="y")
    plt.plot(range(off + 1, min(end + 1, len(perplexities) + 1)), moving_avg[off:end])

    plt.savefig(os.path.join(lpfolder, fname))
    plt.close()


def plot_auc_ap(
    train_aucs, train_aps, test_aucs, test_aps, run_name, step_num, test_every=250, run_folder=""
):

    fname = f"{run_name}_AUCs_APs_{step_num:05d}"
    aucfolder = osp.join(run_folder, "auc_plots")
    if not osp.exists(aucfolder) and not osp.isdir(aucfolder):
        os.mkdir(aucfolder)

    num_vals = len(test_aps)  # They'll all be the same length

    # Set where to start and how much smoothing
    if num_vals < 3:  # Skip first plot, boring
        return
    elif num_vals <= 10:
        st = 0
        smooth = 0
    elif num_vals <= 20:
        st = 3
        smooth = 2
    elif num_vals <= 30:
        st = 5
        smooth = 3
    elif num_vals <= 40:
        st = 7
        smooth = 4
    elif num_vals <= 50:
        st = 10
        smooth = 5
    else:
        st = num_vals // 3
        smooth = 10

    x_vals = [(i + 1 + st) * test_every for i in range(len(test_aps) - st - 1)]
    if len(x_vals) > len(test_aucs) - (st + 1):
        x_vals = x_vals[:-1]

    vals_1 = train_aucs[st + 1 :]
    vals_2 = train_aps[st + 1 :]
    vals_3 = test_aucs[st + 1 :]
    vals_4 = test_aps[st + 1 :]

    if smooth > 0:
        vals_1 = uniform_filter1d(vals_1, size=smooth)
        vals_2 = uniform_filter1d(vals_2, size=smooth)
        vals_3 = uniform_filter1d(vals_3, size=smooth)
        vals_4 = uniform_filter1d(vals_4, size=smooth)

    plt.figure(figsize=(15, 7))
    plt.title(f"Train/test AUC/AP for GVQVAE at step {step_num}")
    min_y = np.min(train_aucs[st:] + train_aps[st:] + test_aucs[st:] + test_aps[st:])
    plt.ylim(min_y, 1.00)

    plt.plot(x_vals, vals_1)
    plt.plot(x_vals, vals_2)
    plt.plot(x_vals, vals_3)
    plt.plot(x_vals, vals_4)

    plt.legend(["Train AUC", "Train AP", "Test AUC", "Test AP"])
    plt.grid(axis="y")
    plt.savefig(osp.join(aucfolder, fname))
    plt.close()
