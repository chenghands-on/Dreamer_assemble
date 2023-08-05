import math
import numpy as np
import re
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch import distributions as torchd

from ..tools_v3 import *
from .. math_functions import *

# This is a work-in-progress attempt to use type aliases to indicate the shapes of tensors.
# T = 50         (TBTT length)
# B = 50         (batch size)
# I = 1/3/10     (IWAE samples)
# A = 3          (action dim)
# E              (embedding dim)
# F = 2048+32    (feature_dim)
# H = 10         (dream horizon)
# J = H+1 = 11
# M = T*B*I = 2500
TensorTBCHW = Tensor
TensorTB = Tensor
TensorTBE = Tensor
TensorTBICHW = Tensor
TensorTBIF = Tensor
TensorTBI = Tensor
TensorJMF = Tensor
TensorJM2 = Tensor
TensorHMA = Tensor
TensorHM = Tensor
TensorJM = Tensor

IntTensorTBHW = Tensor
StateB = Tuple[Tensor, Tensor]
StateTB = Tuple[Tensor, Tensor]


class MLP_v2(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, layer_norm, activation=nn.ELU):
        super().__init__()
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        dim = in_dim
        for i in range(hidden_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation()
            ]
            dim = hidden_dim
        layers += [
            nn.Linear(dim, out_dim),
        ]
        if out_dim == 1:
            layers += [
                nn.Flatten(0),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y
    
class MLP_v3(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm="LayerNorm",
        dist="normal",
        std=1.0,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
    ):
        super(MLP_v3, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, units, bias=False))
            layers.append(norm(units, eps=1e-03))
            layers.append(act())
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "normal":
            return ContDist(
                torchd.independent.Independent(
                    torchd.normal.Normal(mean, std), len(shape)
                )
            )
        if dist == "huber":
            return ContDist(
                torchd.independent.Independent(
                    UnnormalizedHuber(mean, std, 1.0), len(shape)
                )
            )
        if dist == "binary":
            return Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        if dist == "symlog_disc":
            return DiscDist(logits=mean, device=self._device)
        if dist == "symlog_mse":
            return SymlogDist(mean)
        raise NotImplementedError(dist)


class ActionHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        size,
        layers,
        units,
        act=nn.ELU,
        norm=nn.LayerNorm,
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        max_std=1.0,
        temp=0.1,
        outscale=1.0,
        unimix_ratio=0.01,
    ):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, eps=1e-03))
            pre_layers.append(act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            mean = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x



class NoNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class CategoricalSupport(D.Categorical):

    def __init__(self, logits, sup):
        assert logits.shape[-1:] == sup.shape
        super().__init__(logits=logits)
        self.sup = sup

    @property
    def mean(self):
        return torch.einsum('...i,i->...', self.probs, self.sup)
