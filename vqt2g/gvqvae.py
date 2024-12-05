
import logging
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

from typing import Optional, Tuple

from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import VGAE
from torch_geometric.utils import add_self_loops, remove_self_loops, batched_negative_sampling

from vqt2g.vq_layer import NoVQBottleneck
from mpl_toolkits.axes_grid1 import make_axes_locatable
_LOG = logging.getLogger("vqt2g_logger")


class GVQVAE(VGAE):
    def __init__(
        self,
        encoder: Module,
        decoder: Module,
        vq_bottleneck: Module,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float,
        codebook_init_sd: float = 0.05,
    ):
        super().__init__(encoder, decoder)
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self._EPS = 1e-15  # Ensures non-zero vals in torch.log()
        self.vq = vq_bottleneck(codebook_size, embedding_dim, commitment_cost, codebook_init_sd)
        self.max_nodes = self.decoder.max_nodes
        self.codes_per_graph = self.encoder.codes_per_graph
        self.initial_rand_feat_sd = self.encoder.random_feature_sd

    @property
    def codebook(self) -> Tensor:
        """The model's codebook tensor"""
        if isinstance(self.vq.embedding, NoVQBottleneck):
            return torch.zeros((self.codebook_size, self.embedding_dim))
        return self.vq.embedding.weight.data

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor, batch: Tensor) -> Tensor:
        pos_decoded = self.decoder(z, pos_edge_index)
        pos_loss = -torch.log(pos_decoded + self._EPS).mean()
        # Don't want self-loops in negative samples, exclude them
        pos_edge_index, _ = remove_self_loops(pos_edge_index)  # Ignore edge features (second term)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        num_neg = int(pos_edge_index.size(1))  # This is the default anyway
        neg_edge_index = batched_negative_sampling(pos_edge_index, batch, num_neg_samples=num_neg)

        neg_decoded = self.decoder(z, neg_edge_index)
        neg_loss = -torch.log(1 - neg_decoded + self._EPS).mean()

        return pos_loss + neg_loss

    def compute_auc_ap(
        self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor
    ) -> Tuple[float, float]:
        pos_y, neg_y = z.new_ones(pos_edge_index.size(1)), z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred, neg_pred = self.decoder(z, pos_edge_index), self.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def vq_encode(self, only_return_ids: bool = False, *args, **kwargs) -> Tensor:
        """Run encoder then VQ, return embedding vectors"""
        z = self.encode(*args, **kwargs)
        if only_return_ids:
            return self.vq(z, only_return_ids=True)
        _, embs, _, _ = self.vq(z)
        return embs

    @torch.no_grad()
    def encode_graph(
        self,
        graph,
        to_codebook_vectors=True,
        to_numpy=True,
        device=torch.device("cuda:0"),
        to_ids=False,
    ):

        self.eval()
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        if to_codebook_vectors:
            encoded_graph = self.vq_encode(x=x, edge_index=edge_index, only_return_ids=to_ids)
        else:
            encoded_graph = self.encode(x=x, edge_index=edge_index)
        return encoded_graph.cpu().numpy() if to_numpy else encoded_graph

    @torch.no_grad()
    def decode_from_tensor(self, z, num_nodes=None, device=torch.device("cuda:0")):
        """Generate edge probabilities for all edge pairs, from tensor of latent vectors
        Assumes `z` only contains codebook vectors.
        """
        if num_nodes is None:
            num_nodes = self.max_nodes
        pairs = self._all_edges(num_nodes).to(device)
        self.eval()
        return self.decoder(z=z, edge_index=pairs)

    def decode_from_codes(self, codes, num_nodes=None, device=torch.device("cuda:0")):
        """Generate edge probabilities for all edge pairs, from tensor of codebook ids"""
        z = self.code2vec(codes).to(device)
        return self.decode_from_tensor(z=z, num_nodes=num_nodes, device=device)

    @torch.no_grad()
    def encode_decode_graph(self, graph, num_nodes=None, device=torch.device("cuda:0")):
        """Encode graph, VQ, then decode. Return edge probabilities"""
        if num_nodes is None:
            num_nodes = self.max_nodes
        elif num_nodes == "same":  # Force same num nodes in reconstruction
            num_nodes = self.real_num_nodes(graph)
        z = self.encode_graph(graph, to_numpy=False, device=device)
        return self.decode_from_tensor(z, num_nodes=num_nodes, device=device)

    def code2vec(self, ids: Tensor) -> Tensor:
        """Codebook IDs to codebook vectors"""
        device = self.vq.embedding.weight.device
        ids = ids.to(device)
        encodings = torch.zeros(ids.shape[0], self.vq.codebook_size, device=device)
        encodings.scatter_(1, ids.unsqueeze(0), 1)
        return torch.matmul(encodings, self.vq.embedding.weight)

    def vec2code(self, vecs: Tensor) -> Tensor:
        """Latent vectors to (closest) codebook ids"""
        flat_input = vecs.view(-1, self.embedding_dim)
        return self.vq.input_to_ids(flat_input)

    # def _all_edges(self, num_nodes: int) -> Tensor:
    #     """Get tensor of all lower-tri edge index pairs for `num_nodes` nodes
    #        Generates edges for undirected graphs (lower-triangular indices) """
    #     edge_index = [(i, j) for i in range(1, num_nodes) for j in range(i)]
    #     return torch.tensor(edge_index).transpose(1, 0)
    
    def _all_edges(self, num_nodes: int) -> Tensor:
        edge_index = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        return torch.tensor(edge_index).transpose(1, 0)


    def real_num_nodes(self, graph):
        """Number of nodes in graph, excluding padding nodes"""
        return self.max_nodes - int(graph.x[:, -1].sum().item())

    @property
    def rand_feat_sd(self):
        """Current SD of random features"""
        return self.encoder.random_feature_sd

    def stop_rand_feats(self):
        """Stop using random features, use zeros in those feats instead"""
        self.encoder.random_feature_sd = 0.0

    def set_rand_feat_sd(self, value: Optional[float] = 0.0):
        """Set random feature SD back to its initial value or another value"""
        if value is None:  # Reset back to the value it was first set to
            self.encoder.random_feature_sd = self.initial_rand_feat_sd
        else:
            self.encoder.random_feature_sd = value
    
    
    @torch.no_grad()
    def plot_codebook(self, graphs, fname, dims=(0, 1), step_num=0, cpg=100):

        if not fname.endswith(".png"):
            fname += ".png"

        max_points = 50000
        max_graphs = int(np.ceil(max_points / cpg))  # Num graphs to encode

        colx, coly = dims
        self.eval()
        enc_vals = [
            self.encode_graph(graph, to_numpy=False, to_codebook_vectors=False).tolist()
            for graph in graphs[:max_graphs]
        ]
        enc_vals = torch.tensor(enc_vals).cpu()
        x = enc_vals[:, :, colx].view(-1)
        y = enc_vals[:, :, coly].view(-1)
        code_idx = list(range(self.codes_per_graph)) * min(
            len(graphs), max_graphs
        )  # code/node indices
        codebook_x, codebook_y = self.codebook[:, colx].cpu(), self.codebook[:, coly].cpu()

        lim = torch.abs(torch.cat((x, y, codebook_x, codebook_y))).max().item() * 1.05
        colormap = cm.viridis

        # Create the plot
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.set_title(f"Encoder and codebook embeddings for dims {colx} and {coly}, step {step_num}")
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_color("none")
        ax.set_xlim((-lim, lim))
        ax.set_ylim((-lim, lim))

        # Scatter plot for encoder embeddings
        scatter = ax.scatter(x, y, c=code_idx, s=15, marker="o", alpha=0.4, cmap=colormap)

        # Scatter plot for codebook embeddings
        ax.scatter(codebook_x, codebook_y, c="red", s=60, marker="^", alpha=0.25)

        # Add a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

        # Save the plot
        plt.savefig(fname)
        plt.close(fig)