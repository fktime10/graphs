import os
import time
import typer
from pathlib import Path
from vqt2g.run_vqt2g import (
    VQT2GConfig,
    setup_new_run,
    gvqvae_runner,
)
from vqt2g.utils.utils import set_seeds, start_logger

app = typer.Typer()

gvqvae_config = typer.Argument(..., help="Config file for GVQVAE training")
gvqvae_log_level = typer.Option("INFO", help="Python logging level")
gvqvae_comment = typer.Option(None, help="Comment for this GVQVAE run")

@app.command()
def gvqvae_train(
    config_file: Path = gvqvae_config,
    log_level: str = gvqvae_log_level,
    comment: str = gvqvae_comment,
):
    """Train the GVQVAE"""
    config = VQT2GConfig(config_file)
    set_seeds(config.config.seed)
    setup_new_run(config)

    log_file = Path(config.config.this_run_dir, "log_gvqvae.txt")
    start_logger(log_file, log_level)

    if comment is not None:
        comment_file = Path(config.config.this_run_dir, "gvqvae_comment.txt")
        with open(comment_file, "w") as f:
            f.write(comment)

    gvqvae_runner(config)

if __name__ == "__main__":
    app()
