import logging
import os
import pickle as pkl
import random

import numpy as np
import torch

from datetime import datetime as dt
from pathlib import Path

from torch_geometric.loader import DataLoader
from torch_geometric.utils import batched_negative_sampling

from vqt2g.utils.gvqvae_utils import (
    codebook_check,
    codebook_refit,
    dt_fmt,
    save_gvqvae,
    plot_losses,
    plot_auc_ap,
)

_LOG = logging.getLogger("vqt2g_logger")


def gvqvae_train(model, optimizer, device, graph_batch):
    model.train()
    optimizer.zero_grad()

    # Get the 3 parts of the batch and put onto device
    node_feats = graph_batch.x.to(device)
    pos_edge_index = graph_batch.edge_index.to(device)
    batch = graph_batch.batch.to(device)

    # Encode batch and compute vq loss
    z = model.encode(node_feats, pos_edge_index, batch)
    vq_loss, _, perplexity, _ = model.vq(z)

    # model.recon_loss() calls model.decoder()
    # The encoder output `z` is not mapped to nearest codebook vectors here
    recon_loss = model.recon_loss(z, pos_edge_index, batch)
    kl_loss = model.kl_loss()
    if random.random() < 0.002:
        _LOG.debug(
            f"Recon loss only: {recon_loss:.05f}, KL loss raw: {kl_loss:.05f}, "
            f"KL loss scaled: {(1 / graph_batch.num_nodes) * kl_loss:.05f}, "
            f"Batch num nodes: {graph_batch.num_nodes}"
        )
    recon_loss = recon_loss + (1 / graph_batch.num_nodes) * kl_loss

    loss = recon_loss + vq_loss
    loss.backward()
    optimizer.step()

    return loss.item(), recon_loss.item(), perplexity.item(), vq_loss.item()


@torch.no_grad()
def gvqvae_test(model, loader, device, max_items=0, return_means=True):
    model.eval()
    aucs, aps = [], []
    for idx, graph_batch in enumerate(loader):
        if max_items > 0 and idx > np.ceil(max_items / loader.batch_size):
            break

        x = graph_batch.x.to(device)
        batch = graph_batch.batch.to(device)
        pos_edge_index = graph_batch.edge_index.to(device)
        neg_edge_index = batched_negative_sampling(pos_edge_index, batch).to(device)
        z = model.vq_encode(x=x, edge_index=pos_edge_index, batch=batch)

        auc, ap = model.compute_auc_ap(z, pos_edge_index, neg_edge_index)
        aucs.append(auc)
        aps.append(ap)
    if return_means:
        return np.mean(aucs), np.mean(aps)
    else:
        return aucs, aps


def train_autoencoder(
    config,
    model,
    train_graphs,
    test_graphs,
    optimizer,
    scheduler,
):
    """Run the autoencoder training"""

    model_fname = config.config.run_name
    epochs = config.gvqvae.train.epochs
    batch_size = config.gvqvae.train.batch_size
    device = config.config.device

    test_every = config.gvqvae.train.test_every_steps
    eval_train_samples = config.gvqvae.train.max_test_samples

    plot_codebook = config.gvqvae.train.do_embedding_plots
    plot_fname = str(Path(config.gvqvae.plots_dir, config.config.run_name))
    codes_per_graph = config.gvqvae.model.codes_per_graph

    codebook_check_samples = config.gvqvae.train.codebook_check_samples
    codebook_refit_steps = config.gvqvae.train.codebook_refit_steps
    codebook_refit_samples = config.gvqvae.train.codebook_refit_samples

    model_save_every = config.gvqvae.train.checkpoint_steps
    ckpt_dir = config.gvqvae.checkpoint_dir
    cancel_if_loss_spike_after = 1000  ### put in config?

    # Set step number if resuming from checkpoint
    resuming = config.gvqvae.train.resume_training
    step_num = config.gvqvae.train.resume_step if resuming else 0

    _LOG.debug(
        f"model_fname={model_fname}, batch_size={batch_size}, device={device}, test_every={test_every}\n"
        f"num_train={len(train_graphs)}, num_test={len(test_graphs)}, eval_samples={eval_train_samples}\n"
        f"num_code_check={codebook_check_samples}, refit_steps={codebook_refit_steps}, refit_samples={codebook_refit_samples}\n"
        f"save_every={model_save_every}, save_loc={ckpt_dir}"
    )

    # Regular data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    # Separate loader for mid-epoch testing
    tr_eval_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    train_start, steps_start = dt.now(), dt.now()
    train_aucs, train_aps, test_aucs, test_aps = [], [], [], []
    total_losses, recon_losses, perplexities, vq_losses = [], [], [], []
    break_epoch = False

    steps_per_epoch = int(np.ceil(len(train_graphs) / batch_size))
    total_train_steps = steps_per_epoch * epochs
    _LOG.info("Starting GVQVAE training")
    _LOG.info(
        f"Total train steps: {total_train_steps}, num epochs: {epochs}, steps per epoch: {steps_per_epoch}\n"
    )

    # Plot in some early steps
    early_plot_steps = set([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90])

    for epoch in range(1, epochs + 1):
        for idx, graph_batch in enumerate(train_loader):
            step_num += 1
            total_loss, recon_loss, perplexity, vq_loss = gvqvae_train(
                model=model,
                optimizer=optimizer,
                device=device,
                graph_batch=graph_batch,
            )
            total_losses.append(total_loss)
            recon_losses.append(recon_loss)
            perplexities.append(perplexity)
            vq_losses.append(vq_loss)

            # Model eval
            if step_num % test_every == 0 or step_num == total_train_steps:
                test_start = dt.now()

                train_auc_list, train_ap_list = gvqvae_test(
                    model=model,
                    loader=tr_eval_loader,
                    device=device,
                    max_items=eval_train_samples,
                    return_means=False,
                )
                test_auc_list, test_ap_list = gvqvae_test(
                    model=model,
                    loader=test_loader,
                    device=device,
                    max_items=eval_train_samples,
                    return_means=False,
                )
                train_auc = np.mean(train_auc_list)
                train_ap = np.mean(train_ap_list)
                test_auc = np.mean(test_auc_list)
                test_ap = np.mean(test_ap_list)
                train_aucs.append(train_auc)
                train_aps.append(train_ap)
                test_aucs.append(test_auc)
                test_aps.append(test_ap)

                # Mean train losses since previous test
                mean_loss = np.mean(total_losses[-test_every:])
                mean_recon_loss = np.mean(recon_losses[-test_every:])
                mean_perplexity = np.mean(perplexities[-test_every:])
                mean_vq_loss = np.mean(vq_losses[-test_every:])

                steps_time = str(dt.now() - steps_start).split(".")[0][2:]
                total_time = dt_fmt(dt.now() - train_start)
                test_time = str(dt.now() - test_start)[3:-4]
                current_lr = scheduler.optimizer.param_groups[0]["lr"]

                _LOG.info(
                    f"Step {step_num:5d} (epoch {epoch}) mean (tr, te) AUCs: {train_auc:.06f}, "
                    f"{test_auc:.06f} | APs: {train_ap:.06f}, {test_ap:.06f}"
                )
                _LOG.info(
                    f"Loss  : {mean_loss:.05f}, Recon loss: {mean_recon_loss:.06f}, Perplexity: "
                    f"{mean_perplexity:.04f}, VQ loss: {mean_vq_loss:.06f}"
                )

                _LOG.debug(
                    f"TRAIN AUC/AP min/max: {min(train_auc_list):.04f} - {max(train_auc_list):.04f}, {min(train_ap_list):.04f} - {max(train_ap_list):.04f}"
                )
                _LOG.debug(
                    f"TEST  AUC/AP min/max: {min(test_auc_list):.04f} - {max(test_auc_list):.04f}, {min(test_ap_list):.04f} - {max(test_ap_list):.04f}"
                )
                _LOG.info(
                    f"Since last test: {steps_time},   total {total_time}   ||   Test took: "
                    f"{test_time}   ||   Current LR: {current_lr:.4e}"
                )

                # Found decaying on test AUC more reliable than decay on loss
                scheduler.step(test_auc)

                steps_start = dt.now()

            # Plot the codebook
            if plot_codebook and (step_num in early_plot_steps or step_num % test_every == 0):
                fname = f"{plot_fname}_{step_num:06d}"
                model.plot_codebook(
                    train_graphs, fname=fname, dims=(0, 1), step_num=step_num, cpg=codes_per_graph
                )

            # Plot losses and AUC/APs every second test
            if step_num % (test_every * 2) == 0:
                _LOG.debug(f"Plotting losses for step {step_num:06d}")
                plot_losses(
                    total_losses,
                    recon_losses,
                    vq_losses,
                    perplexities,
                    model_fname,
                    step_num,
                    config.gvqvae.gvqvae_dir,
                )
                plot_auc_ap(
                    train_aucs,
                    train_aps,
                    test_aucs,
                    test_aps,
                    model_fname,
                    step_num,
                    test_every,
                    config.gvqvae.gvqvae_dir,
                )

            # Check codebook usage
            if test_every > 0 and step_num % test_every == 0:
                codebook_check(
                    model=model,
                    codebook_dataset=train_graphs,
                    device=device,
                    step_num=step_num,
                    max_codebook_samples=codebook_check_samples,
                )

            # Refit codebook using k-means on encoder outputs
            if codebook_refit_steps is not None and step_num in codebook_refit_steps:
                _LOG.info(f"Saving checkpoint, then refitting codebook at step {step_num}")
                ckpt_fname = f"{model_fname}_gvqvae_step_{step_num:06d}_before_refit.pt"
                save_gvqvae(config, ckpt_dir, ckpt_fname, model, step_num, optimizer, scheduler)
                codebook_refit(
                    model=model,
                    dataset=train_graphs,
                    device=device,
                    kmeans_dataset_size=codebook_refit_samples,
                    step_num=step_num,
                )
                fname = f"{plot_fname}_post-refit_{step_num:06d}"
                model.plot_codebook(train_graphs, fname=fname, dims=(0, 1), step_num=step_num)

            # Save the model + loss logs
            if model_save_every > 0 and step_num % model_save_every == 0:
                ckpt_fname = f"{model_fname}_gvqvae_step_{step_num:06d}.pt"
                losses_full_fname = f"losses_{model_fname}_gvqvae.pkl"

                save_gvqvae(config, ckpt_dir, ckpt_fname, model, step_num, optimizer, scheduler)

                all_losses = [total_losses, recon_losses, perplexities, vq_losses]
                with open(os.path.join(ckpt_dir, losses_full_fname), "wb") as f:
                    pkl.dump(all_losses, f)

            # If loss spikes assume the run is dead and quit
            latest_losses_avg = np.mean(total_losses[-20:])
            if (
                cancel_if_loss_spike_after > 0
                and step_num > cancel_if_loss_spike_after
                and latest_losses_avg > 10
            ):
                _LOG.error(
                    f"Loss spike at step {step_num}: {round(max(total_losses[-20:]), 4)}. Cancelling run"
                )
                break_epoch = True
                break
        if break_epoch:
            break
    # Save final model. Use different filename to distinguish from checkpoints
    final_fname = f"{model_fname}_gvqvae_final.pt"
    losses_full_fname = f"losses_{model_fname}_step_{step_num}_final.pkl"
    aucs_aps_fname = f"aucs_aps_{model_fname}_step_{step_num}_final.pkl"

    save_gvqvae(config, ckpt_dir, final_fname, model, step_num, optimizer, scheduler)

    all_losses = [total_losses, recon_losses, perplexities, vq_losses]
    aucs_aps = [train_aucs, train_aps, test_aucs, test_aps]

    with open(os.path.join(ckpt_dir, losses_full_fname), "wb") as f:
        pkl.dump(all_losses, f)
    with open(os.path.join(ckpt_dir, aucs_aps_fname), "wb") as f:
        pkl.dump(aucs_aps, f)

    _LOG.info(f"Finished GVQVAE training, took {dt_fmt(dt.now()-train_start)}, {step_num} steps")