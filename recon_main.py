import os
import time
import typer
from pathlib import Path
from vqt2g.run_recons import recon_runner
from vqt2g.run_vqt2g import VQT2GConfig
from vqt2g.utils.utils import set_seeds, start_logger

app = typer.Typer()

recon_config = typer.Argument(..., help="Config file for recon run")
recon_log_level = typer.Option("INFO", help="Python logging level")
recon_comment = typer.Option(None, help="Comment for this run")
recon_threshold = typer.Option(0.5, help="Threshold for edge adding")
recon_num = typer.Option(30, help="Number of graphs from train/test set to use")
recon_full_heatmap = typer.Option(False, help="Include padding nodes in heatmaps")
recon_edge_sampling = typer.Option(False, help="Use sampling instead of top-k for edges")

@app.command()
def run_recon(
    config_file: Path = recon_config,
    num: int = recon_num,
    threshold: float = recon_threshold,
    edge_sampling: bool = recon_edge_sampling,
    heatmap_padding: bool = recon_full_heatmap,
    log_level: str = recon_log_level,
    comment: str = recon_comment,
):
    """Run graph reconstructions using GVQVAE"""
    config = VQT2GConfig(config_file)

    test_dir = Path(config.config.this_run_dir, "recons")
    os.makedirs(test_dir, exist_ok=True)

    set_seeds(config.config.seed)

    eval_fname = f"log_model_recons_{time.strftime('%b_%d_%H-%M')}.txt"
    log_file = Path(test_dir, eval_fname)
    start_logger(log_file, log_level)

    recon_runner(
        config=config,
        num_recon=num,
        threshold=threshold,
        edge_sampling=edge_sampling,
        heatmap_padding=heatmap_padding,
        comment=comment,
    )

if __name__ == "__main__":
    app()

 