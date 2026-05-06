import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MLP, self).__init__()
        
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_size, output_size))
        else:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# based on https://github.com/lyakaap/NetVLAD-pytorch
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()

        self.num_clusters = num_clusters

        self.alpha = alpha
        self.normalize_input = normalize_input

        self.conv = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C, D = x.shape[:]  # batch, num patches, dimension
        x = x.permute(0, 2, 1)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).reshape(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.reshape(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-2), -1, -1).permute(1, 0, 2).unsqueeze(0)

        residual *= soft_assign.unsqueeze(-1)

        vlad = residual.sum(dim=-2)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
    

class NetVLADLayer(nn.Module):
    def __init__(self, dim=768, cluster_num=8, alpha=50.0, T=16,
                 num_mlp_layers=1, mlp_hidden_dim=768, mlp_output_dim=None):
        super().__init__()
        
        self.dim = dim
        self.T = T
        self.effective_dim = self.dim * T
        
        self.net_vlad = NetVLAD(num_clusters=cluster_num, alpha=alpha, dim=self.effective_dim)
        
        if num_mlp_layers > 0:
            input_size = self.effective_dim * cluster_num
            output_size = mlp_output_dim if mlp_output_dim is not None else input_size
            self.projection = MLP(input_size=input_size, hidden_size=mlp_hidden_dim, num_layers=num_mlp_layers, output_size=output_size)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        if len(x.shape) > 3:
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(x.shape[0], x.shape[1], -1)
        else:
            N, C, D = x.shape[:]  # batch, num patches, dimension
            C = C // self.T
            D = D * self.T
            x = x.reshape(N, C, D)
        vlad_features = self.net_vlad(x)
        projection_output = self.projection(vlad_features)
        return projection_output