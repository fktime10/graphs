"""Encoder and decoder for GVQVAE"""

import torch
import torch.nn as nn
import torch_geometric as pyg

from typing import Optional

from torch import Tensor
from torch_geometric.nn import GCNConv, SAGEConv

_conv_type_map = {
    "gcnconv": GCNConv,
    "sageconv": SAGEConv,
}


class GVQEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels_1: int,
        hid_channels_2: int,
        output_dim: int,
        max_nodes: int,
        codes_per_graph: int = 100,
        conv_type: str = "sageconv",
        conv_aggr: str = "mean",
        num_random_feature: int = 0,
        random_feature_sd: float = 0.5,
        random_feature_only: bool = False,
        pre_vq_batchnorm: bool = True,
        use_linear_layers: bool = True,
        linear_layer_dropout: float = 0.25,
    ):
        """Graph convolutional encoder for GVQVAE"""
        super(GVQEncoder, self).__init__()

        self.use_random_feature = num_random_feature > 0
        self.random_feature_only = random_feature_only
        self.num_random_feature = num_random_feature
        self.random_feature_sd = random_feature_sd

        self.use_linear_layers = use_linear_layers
        self.codes_per_graph = codes_per_graph

        # Get node feature dimension
        if not self.use_random_feature:
            self.num_node_feats = in_channels
        elif random_feature_only:
            self.num_node_feats = num_random_feature
        else:
            self.num_node_feats = in_channels + num_random_feature

        conv_type = conv_type.lower()
        if conv_type not in _conv_type_map:
            raise ValueError(
                f"Invalid GNN layer type: '{conv_type}'. Valid types are "
                f"{list(_conv_type_map.keys())}"
            )
        self.conv_type = _conv_type_map[conv_type]

        self.max_nodes = max_nodes
        self.hid_channels_1 = hid_channels_1
        self.hid_channels_2 = hid_channels_2
        self.output_dim = output_dim
        self.pre_vq_batchnorm = pre_vq_batchnorm

        self.conv1 = self.conv_type(self.num_node_feats, hid_channels_1, aggr=conv_aggr)
        self.bn1 = pyg.nn.BatchNorm(hid_channels_1)
        self.conv2 = self.conv_type(hid_channels_1, hid_channels_2, aggr=conv_aggr)
        self.bn2 = pyg.nn.BatchNorm(hid_channels_2)
        self._conv3_channels = output_dim if self.use_linear_layers else hid_channels_2
        self.conv3 = self.conv_type(hid_channels_2, self._conv3_channels, aggr=conv_aggr)
        self.bn3 = pyg.nn.BatchNorm(self._conv3_channels)

        # Either add two linear layers or one convolution layer
        if self.use_linear_layers:
            self.in_lin_size = self.max_nodes * self._conv3_channels
            self.out_lin_size = self.codes_per_graph * self.output_dim
            self.lin_1 = nn.Sequential(
                nn.Linear(self.in_lin_size, self.out_lin_size),
                nn.Tanh(),
                nn.Dropout(linear_layer_dropout),
            )

            self.lin_mu = nn.Linear(self.out_lin_size, self.out_lin_size)
            self.lin_sigma = nn.Linear(self.out_lin_size, self.out_lin_size)
        else:
            self.conv_x = self.conv_type(hid_channels_2, output_dim, aggr=conv_aggr)
            self.conv_sigma = self.conv_type(hid_channels_2, output_dim, aggr=conv_aggr)

        if self.pre_vq_batchnorm:
            self.bn_final_x = pyg.nn.BatchNorm(output_dim * self.codes_per_graph)
            self.bn_final_sigma = pyg.nn.BatchNorm(output_dim * self.codes_per_graph)

        if not self.use_linear_layers and codes_per_graph != max_nodes:
            raise ValueError(
                "Codes per graph must equal max nodes when not using linear layers in encoder"
            )

    def forward(self, x: Tensor, edge_index: Tensor, batch: Optional[Tensor] = None):
        batch_size = max(batch) + 1 if batch is not None else 1
        x = x.float()

        # Make the random feature tensor during each pass
        if self.use_random_feature:
            if self.random_feature_only:
                x = torch.normal(
                    0, self.random_feature_sd, (x.size(0), self.num_random_feature)
                ).to(x.device)
            else:
                x_rand = torch.normal(
                    0, self.random_feature_sd, (x.size(0), self.num_random_feature)
                ).to(x.device)
                x = torch.cat((x, x_rand), dim=1)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()

        if self.use_linear_layers:
            x = x.view(batch_size, -1)
            x = self.lin_1(x)
            sigma = self.lin_sigma(x)
            x = self.lin_mu(x)
        else:
            sigma = self.conv_sigma(x, edge_index)
            x = self.conv_x(x, edge_index)

        # Node-wise batchnorm
        if self.pre_vq_batchnorm:
            x = x.view(batch_size, self.codes_per_graph * self.output_dim)
            sigma = sigma.view(batch_size, self.codes_per_graph * self.output_dim)
            x = self.bn_final_x(x)
            sigma = self.bn_final_sigma(sigma)

        sigma = sigma.view(-1, self.output_dim)
        x = x.view(-1, self.output_dim)
        return x, sigma


class GVQDecoder(torch.nn.Module):
    def __init__(
        self,
        in_latent_dim: int,
        codes_per_graph: int,
        hidden_size_1: int,
        hidden_size_2: int,
        output_node_dim: int,
        max_nodes: int,
    ):
        """MLP decoder for GVQVAE"""
        super(GVQDecoder, self).__init__()
        self.max_nodes = max_nodes
        self.in_latent_dim = in_latent_dim
        self.codes_per_graph = codes_per_graph
        self.output_node_dim = self.max_nodes * output_node_dim
        self.input_len = self.in_latent_dim * self.codes_per_graph

        self.mlp = nn.Sequential(
            nn.Linear(self.input_len, hidden_size_1),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size_2, hidden_size_2),
            nn.Tanh(),
            nn.Dropout(0.25),
        )
        self.edges_layer = nn.Linear(hidden_size_2, self.output_node_dim)

    def reshape_to_adj(self, adj_vector: Tensor) -> Tensor:
        """Reshape the decoder output (vector) into a symmetric matrix"""
        adj_dims = (self.max_nodes, self.max_nodes)
        mask = torch.zeros(adj_dims)
        tril_inds = torch.tril_indices(self.max_nodes, self.max_nodes, offset=-1)
        mask[tril_inds[0], tril_inds[1]] = adj_vector
        return mask + mask.T

    def forward(self, z: Tensor, edge_index: Tensor):
        batch_size = int(z.shape[0] / self.codes_per_graph)
        z = z.view(-1, self.input_len)
        z = self.mlp(z)
        edge = self.edges_layer(z)
        edge = edge.view((self.max_nodes * batch_size, -1))
        edge_probs = (edge[edge_index[0]] * edge[edge_index[1]]).sum(dim=1)
        edge_probs = torch.sigmoid(edge_probs)
        return edge_probs
