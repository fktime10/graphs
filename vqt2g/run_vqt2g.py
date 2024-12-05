"""Functions to run GVQVAE training"""

import logging
import os
import pickle as pkl
import time

import torch
import yaml

from collections import Counter
from pathlib import Path

from easydict import EasyDict as edict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from vqt2g.encoder_decoder import GVQEncoder, GVQDecoder
from vqt2g.gvqvae import GVQVAE
from vqt2g.load_dataset import load_dataset
from vqt2g.utils.graph_load_batch import graph_load_batch
from vqt2g.utils.gvqvae_utils import load_gvqvae
from vqt2g.vq_layer import VectorQuantizer, VectorQuantizerEMA, NoVQBottleneck
from vqt2g.train_gvqvae import train_autoencoder
_LOG = logging.getLogger("gvqvae_logger")

 
class VQT2GConfig:
    """Class to hold all the config parameters"""

    def __init__(self, config_file):
        """Load configuration from a file"""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        config = edict(config)

        self.config = config
        self.dataset = config.dataset
        self.gvqvae = config.gvqvae

        self.device = config.device

    def as_dict(self):
        """Convert the config to a regular dictionary"""
        return self._as_dict()

    def _as_dict(self, obj=None):
        """Helper method for as_dict"""
        config_dict = {}
        if obj is None:
            obj = self.config

        for key, val in obj.items():
            if isinstance(val, edict):
                config_dict[key] = self._as_dict(obj=val)
            elif isinstance(val, Path):
                config_dict[key] = str(val)
            else:
                config_dict[key] = val

        return config_dict

    def save_config(self, location=".", fname="config.yaml"):
        """Save the config to a yaml file"""
        fpath = Path(self.config.this_run_dir, location, fname)
        with open(fpath, "w") as f:
            yaml.safe_dump(self.as_dict(), f, default_flow_style=False)
        _LOG.info(f"Saved config to: {fpath}")


def setup_new_run(config: VQT2GConfig) -> None:
    """Set up folder for a new experiment run"""
    run_name_short = config.config.run_name
    run_name = f"{run_name_short}_{time.strftime('%b_%d_%H-%M')}"
    config.config["run_name_short"] = run_name_short
    config.config["run_name"] = run_name
    this_run_dir = Path(config.config.run_dir_root, run_name)
    config.config["this_run_dir"] = str(this_run_dir)

    os.makedirs(this_run_dir, exist_ok=True)

    _LOG.info(f"Setting up new run - {this_run_dir}")
    train_test_split_dataset(config)

    # Save the config inside this directory
    config.save_config()


def train_test_split_dataset(config: VQT2GConfig) -> None:
    """Load dataset and perform train/test split"""
    data_dir = Path(config.dataset.dataset_dir, config.dataset.dataset_name)
    config.dataset["dataset_save_path"] = str(data_dir)
    if not data_dir.is_dir():
        os.makedirs(data_dir)

    if config.dataset.is_already_split:
        _LOG.info("Train/test split already done, loading data from file")
        try:
            with open(Path(data_dir, "train_graphs.pkl"), "rb") as f:
                train_graphs = pkl.load(f)
        except FileNotFoundError:
            raise ValueError(
                "Pre-split dataset not found, set `is_already_split = False` in config to fix"
            )
        max_nodes = train_graphs[0].num_nodes
        num_feats = train_graphs[0].x.size(1)
        config.dataset["max_nodes"] = max_nodes
        config.dataset["num_node_features"] = num_feats
        return

    # Load raw dataset
    if config.dataset.dataset_name == "proteins":
        data = graph_load_batch(
            data_dir=config.dataset.raw_dataset,
            min_num_nodes=100,
            max_num_nodes=500,
            name="DD",
        )
    else:
        with open(config.dataset.raw_dataset, "rb") as f:
            data = pkl.load(f)

    # Perform train/test split
    _LOG.info("Loading raw dataset and performing train/test split")
    train_graphs, test_graphs, indices = load_dataset(
        data=data,
        proportion_or_count=config.dataset.prop_or_count,
        test_prop=config.dataset.test_prop,
        test_num=config.dataset.test_num,
        add_node_attrs=config.dataset.add_node_features,
        attr_type=config.dataset.node_feature_type,
        shuffle=config.dataset.shuffle,
        seed=config.config.seed,
        max_dataset_size=config.dataset.max_dataset_size,
        one_hot_degree=config.dataset.degree_feature_one_hot,
    )

    max_nodes = train_graphs[0].num_nodes
    num_feats = train_graphs[0].x.size(1)
    config.dataset["max_nodes"] = max_nodes
    config.dataset["num_node_features"] = num_feats
    _LOG.info(f"Dataset max nodes = {max_nodes}; num node features = {num_feats}")

    # Save each split separately
    with open(Path(data_dir, "train_graphs.pkl"), "wb") as f:
        pkl.dump(train_graphs, f)
    with open(Path(data_dir, "test_graphs.pkl"), "wb") as f:
        pkl.dump(test_graphs, f)
    with open(Path(data_dir, "train_test_indices.pkl"), "wb") as f:
        pkl.dump(indices, f)
    _LOG.info(f"Saved dataset train/test splits to: {data_dir}")


def setup_gvqvae_model(config: VQT2GConfig, device):
    """Initialize the GVQVAE model"""
    cfg_gvq = config.gvqvae
    cfg_model = config.gvqvae.model

    encoder = GVQEncoder(
        in_channels=config.dataset.num_node_features,
        hid_channels_1=cfg_model.encoder_channels_1,
        hid_channels_2=cfg_model.encoder_channels_2,
        output_dim=cfg_model.codebook_dim,
        max_nodes=config.dataset.max_nodes,
        conv_type=cfg_model.gnn_conv_type,
        conv_aggr=cfg_model.gnn_conv_aggr,
        num_random_feature=cfg_model.num_random_feature,
        random_feature_sd=cfg_model.random_feature_sd,
        random_feature_only=cfg_model.random_feature_only,
        pre_vq_batchnorm=cfg_model.pre_vq_batchnorm,
        use_linear_layers=cfg_model.encoder_linear_layers,
        codes_per_graph=cfg_model.codes_per_graph,
        linear_layer_dropout=cfg_model.encoder_dropout,
    )

    _LOG.debug("Created encoder")

    # Decoder
    decoder = GVQDecoder(
        in_latent_dim=cfg_model.codebook_dim,
        codes_per_graph=cfg_model.codes_per_graph,
        hidden_size_1=cfg_model.decoder_size_1,
        hidden_size_2=cfg_model.decoder_size_2,
        output_node_dim=cfg_model.output_node_dim,
        max_nodes=config.dataset.max_nodes,
    )

    _LOG.debug("Created decoder")

    # VQ bottleneck (codebook)
    use_vq = cfg_model.use_vq_bottleneck
    vq_ema = cfg_model.ema
    if not use_vq:
        vq_layer = NoVQBottleneck
    elif vq_ema:
        vq_layer = VectorQuantizerEMA
    else:
        vq_layer = VectorQuantizer
    _LOG.debug(f"Created VQ layer: {vq_layer.__name__}")

    # Full GVQVAE model
    model = GVQVAE(
        encoder=encoder,
        decoder=decoder,
        vq_bottleneck=vq_layer,
        embedding_dim=cfg_model.codebook_dim,
        codebook_size=cfg_model.codebook_size,
        commitment_cost=cfg_gvq.train.commitment_cost,
        codebook_init_sd=cfg_model.codebook_init_sd,
    ).to(device)

    return model


def gvqvae_runner(config: VQT2GConfig) -> None:
    """Run GVQVAE training"""
    cfg_base = config.config
    cfg_gvq = config.gvqvae

    device = torch.device(cfg_base.device)

    gvqvae_dir = Path(cfg_base.this_run_dir, "gvqvae")
    checkpoint_dir = Path(gvqvae_dir, "checkpoints")
    plots_dir = Path(gvqvae_dir, "plots")
    os.makedirs(gvqvae_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if cfg_gvq.train.do_embedding_plots:
        os.makedirs(plots_dir, exist_ok=True)
    cfg_gvq["gvqvae_dir"] = str(gvqvae_dir)
    cfg_gvq["checkpoint_dir"] = str(checkpoint_dir)
    cfg_gvq["plots_dir"] = str(plots_dir)
    config.save_config()

    # Load graph datasets
    data_dir = config.dataset.dataset_save_path
    with open(Path(data_dir, "train_graphs.pkl"), "rb") as f:
        train_graphs = pkl.load(f)
    with open(Path(data_dir, "test_graphs.pkl"), "rb") as f:
        test_graphs = pkl.load(f)

    _LOG.info("Train/test graphs loaded")
    _LOG.info(f"Num train, test graphs = {len(train_graphs)}, {len(test_graphs)}")

    # Create model
    model = setup_gvqvae_model(config, device)
    _LOG.info("GVQVAE initialized")

    # Create optimizer
    use_vq = cfg_gvq.model.use_vq_bottleneck
    lr = cfg_gvq.train.learning_rate
    codebook_lr = lr * cfg_gvq.train.codebook_lr_factor
    weight_decay = cfg_gvq.train.weight_decay
    if use_vq and cfg_gvq.train.codebook_lr_factor != 1.0:
        optimizer = torch.optim.Adam(
            [
                {"params": model.encoder.parameters()},
                {"params": model.decoder.parameters()},
                {"params": model.vq.parameters(), "lr": codebook_lr},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )
        _LOG.info(f"Encoder+decoder learning rate = {lr}, codebook learning rate = {codebook_lr}")
    elif use_vq:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        _LOG.info(f"Encoder+decoder+codebook learning rate = {lr}")
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": model.encoder.parameters()},
                {"params": model.decoder.parameters()},
                {"params": model.vq.parameters(), "lr": 0.0},
            ],
            lr=lr,
            weight_decay=weight_decay,
        )
        _LOG.info(f"Encoder+decoder learning rate = {lr}, no codebook")

    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=cfg_gvq.train.lr_decay_factor,
        patience=cfg_gvq.train.lr_decay_patience,
        threshold=1e-5,
        cooldown=0,
        verbose=True,
        min_lr=cfg_gvq.train.min_lr,
    )

    _LOG.info("Optimizer and scheduler initialized")

    # If resuming training
    if cfg_gvq.train.resume_training:
        load_gvqvae(config, model, device, optimizer, scheduler)
        _LOG.info(f"Loaded checkpoint from {config.gvqvae.test.last_checkpoint}")

    train_autoencoder(
        config=config,
        model=model,
        train_graphs=train_graphs,
        test_graphs=test_graphs,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    config.save_config("gvqvae")
    config.save_config()
    _LOG.info("Training finished and config saved")
