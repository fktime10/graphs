"""Assorted utility functions for running GVQVAE experiments"""

import logging
import random
import numpy as np
import torch


def set_seeds(seed):
    """Set random, numpy and torch seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def start_logger(log_file, level="INFO"):
    """Set up logging that logs to both log file and stderr"""
    level = level.upper()

    logging.root.handlers = []
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(filename)-20s L%(lineno)-3d | %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, "a"), logging.StreamHandler()],
    )

    # When log level is DEBUG these packages can spam logs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return logging.getLogger("gvqvae_logger")
