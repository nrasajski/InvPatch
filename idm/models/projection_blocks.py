import torch
from transformers import GPT2LMHeadModel
import pickle
import torch.nn as nn
from typing import Tuple, Any
from string import punctuation
import torch.nn.functional as F

# Code adapted from https://github.com/YichaoCai1/CLAP/blob/master/main/networks.py
class ResBlock(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, activation, drop_rate=0.5, repeat=0, scale=1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, latent_dim, bias=True)        
        self.activation = activation()
        self.dropout = nn.Dropout(p=drop_rate)
        self.projection = self.__zero_initial(nn.Linear(latent_dim, out_dim, bias=False))
        self.scale = scale
        
        self.repeat = repeat
        if repeat > 0:
            self.latent_linear = nn.Sequential(*[nn.Sequential(*[activation(), nn.Linear(latent_dim, latent_dim, bias=True)])
                                                 for _ in range(repeat)])
        self.out_dim = out_dim
        self.down_sample = not (in_dim == out_dim)
    
    def __zero_initial(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module
    
    def __downsample_nn(self, x):
        bs = x.shape[0]
        x = x.unsqueeze(1)
        x = F.interpolate(x, self.out_dim, mode='nearest')
        return x.view([bs, self.out_dim])
    
    def forward(self, x):
        out = self.activation(x)
        out = self.dropout(out)
        out = self.linear(out)
        if self.repeat > 0:
            out = self.latent_linear(out)
        out = self.projection(out)
        out = out * self.scale
        if self.down_sample:
            x = self.__downsample_nn(x)
        return out + x     

class ResMLP(nn.Module):
    def __init__(self,
                 sizes: Tuple[int, ...],
                 activation=nn.Tanh,
                 drop_rate=0.5) -> None:
        super().__init__()
   
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.LayerNorm(sizes[i]))
            layers.append(ResBlock(in_dim=sizes[i], latent_dim=sizes[i + 1], out_dim=sizes[i + 1], repeat=0,
                        activation=activation, drop_rate=drop_rate))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MeanResMLP(nn.Module):
    def __init__(self,
                 sizes: Tuple[int, ...],
                 activation=nn.Tanh,
                 drop_rate=0.5) -> None:
        super().__init__()
   
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.LayerNorm(sizes[i]))
            layers.append(ResBlock(in_dim=sizes[i], latent_dim=sizes[i + 1], out_dim=sizes[i + 1], repeat=0,
                        activation=activation, drop_rate=drop_rate))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.mean(1)
        return self.model(x)

class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class MeanMLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(1)
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MeanMLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.LayerNorm(sizes[i]))
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class MeanEverything(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(1)