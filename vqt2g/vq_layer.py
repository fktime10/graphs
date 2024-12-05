
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import MiniBatchKMeans
from torch import Tensor

_LOG = logging.getLogger("vqt2g_logger")


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float,
        codebook_init_sd: float = 1.0,
    ):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        self.init_codebook(codebook_init_sd)

    def init_codebook(self, codebook_init_sd: float) -> None:
        """Initialise/re-initialise the codebook with random normals"""
        self.embedding.weight.data.normal_(0, codebook_init_sd)

    def input_to_ids(self, flat_input: Tensor) -> Tensor:
        """Convert flattened input to codebook ids (ints)"""
        # Calculate distances
        input_sq_sum = torch.sum(flat_input ** 2, dim=1, keepdim=True)
        emb_wt_sq_sum = torch.sum(self.embedding.weight ** 2, dim=1)
        emb_wt_t = self.embedding.weight.t()
        distances = input_sq_sum + emb_wt_sq_sum - 2 * torch.matmul(flat_input, emb_wt_t)

        # `encoding_indices`: vector of integers (codebook IDs)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices

    def refit_codebook_kmeans(
        self, encoded_nodes: np.ndarray, batch_size: int = 4096, max_iter: int = 1000
    ) -> None:

        assert len(encoded_nodes) > self.codebook_size
        device = self.embedding.weight.device

        k_means_model = MiniBatchKMeans(
            n_clusters=self.codebook_size,
            max_iter=max_iter,
            batch_size=max_iter,
            verbose=0,
            compute_labels=False,  # don't fit the train data labels after training
            max_no_improvement=100,
        )
        k_means_model = k_means_model.fit(encoded_nodes)
        centres = torch.tensor(k_means_model.cluster_centers_, device=device, dtype=torch.float32)
        # Set embedding weight to the new cluster centres
        self.embedding.weight.data = centres

    def forward(self, inputs: Tensor, only_return_ids: bool = False):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        encoding_indices = self.input_to_ids(flat_input)

        if only_return_ids:
            return encoding_indices

        # One-hot matrix indicating codebook id
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Look up the codebook latent vectors
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return vq_loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        epsilon: float = 1e-5,
        codebook_init_sd: float = 1.0,
    ):
        super(VectorQuantizerEMA, self).__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        # In non-EMA this starts at uniform(+/- 1/self._num_embeddings)
        self.init_codebook(codebook_init_sd)

        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.ema_w = nn.Parameter(torch.Tensor(codebook_size, self.embedding_dim))
        self.ema_w.data.normal_()

        self.decay = decay
        self.epsilon = epsilon

    def init_codebook(self, codebook_init_sd: float) -> None:
        """Initialise/re-initialise the codebook with random normals"""
        self.embedding.weight.data.normal_(0, codebook_init_sd)

    def input_to_ids(self, flat_input: Tensor) -> Tensor:
        """Convert flattened input to codebook ids (ints)"""
        # Calculate distances
        input_sq_sum = torch.sum(flat_input ** 2, dim=1, keepdim=True)
        emb_wt_sq_sum = torch.sum(self.embedding.weight ** 2, dim=1)
        emb_wt_t = self.embedding.weight.t()
        distances = input_sq_sum + emb_wt_sq_sum - 2 * torch.matmul(flat_input, emb_wt_t)

        # `encoding_indices`: vector of integers (codebook IDs)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices

    def ema_update(self, encodings: Tensor, flat_input: Tensor) -> None:
        """Do the EMA update step during training"""
        cluster_decay = self.ema_cluster_size * self.decay
        cluster_update = (1 - self.decay) * torch.sum(encodings, 0)
        self.ema_cluster_size = cluster_decay + cluster_update

        # Laplace smoothing of the cluster size
        n = torch.sum(self.ema_cluster_size.data)
        self.ema_cluster_size = (
            (self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
        )

        # Update embeddings
        dw = torch.matmul(encodings.t(), flat_input)
        self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)

        self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))

    def refit_codebook_kmeans(
        self, encoded_nodes: np.ndarray, batch_size: int = 4096, max_iter: int = 1000
    ) -> None:
        assert len(encoded_nodes) > self.codebook_size
        device = self.embedding.weight.device

        k_means_model = MiniBatchKMeans(
            n_clusters=self.codebook_size,
            max_iter=max_iter,
            batch_size=max_iter,
            verbose=0,
            compute_labels=False,  # don't fit the train data labels after training
            max_no_improvement=100,
        )
        k_means_model = k_means_model.fit(encoded_nodes)
        centres = torch.tensor(k_means_model.cluster_centers_, device=device, dtype=torch.float32)
        # Set embedding weight to the new cluster centres
        self.embedding.weight.data = centres

    def forward(self, inputs: Tensor, only_return_ids: bool = False):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        encoding_indices = self.input_to_ids(flat_input)

        if only_return_ids:
            return encoding_indices

        # Encodings: One-hot matrix indicating codebook id
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Look up the codebook latent vectors
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors while training
        if self.training:
            self.ema_update(encodings, flat_input)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return vq_loss, quantized, perplexity, encodings


class NoVQBottleneck(nn.Module):
    # Keeping these args but they don't do anything
    def __init__(
        self, codebook_size: int = 0, embedding_dim: int = 0, commitment_cost: float = 0.0
    ):
        super(NoVQBottleneck, self).__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = 1
        self.commitment_cost = 0
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        # This way probably better than ^^ but code later on fails. Maybe fix later
        # self.register_buffer("embedding", torch.zeros((self.codebook_size, self.embedding_dim)))

    def forward(self, inputs, only_return_ids=False):
        if only_return_ids:
            raise ValueError("Can't return codebook ids as NoVQBottleneck doesn't have a codebook")
        # Pretend to return these
        vq_loss = torch.tensor(0)
        perplexity = torch.tensor(0)
        encodings = 0

        # Actually return inputs unchanged
        quantized = inputs
        return vq_loss, quantized, perplexity, encodings