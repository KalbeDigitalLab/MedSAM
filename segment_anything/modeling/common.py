# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from typing import Optional, Tuple, Type


class LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRALayer_qkv_timm(nn.Module):
    """In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class AdapterMLPBlock(MLPBlock):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: type = nn.ReLU) -> None:
        super().__init__(embedding_dim, mlp_dim, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x))) + x


class AdditionAdapterMLPBlock(AdapterMLPBlock):
    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x_1) + x_2)) + x_1


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SuperScalableLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank):
        super(SuperScalableLinear, self).__init__(in_features=in_features, out_features=out_features)
        config_A_B = [f'LoRA_{rank}', 'vector', 'constant', 'none']
        config_C = [f'LoRA_{rank}', 'vector', 'none']
        config_D_E = ['constant', 'none', 'vector']
        self.configs = []
        for A in config_A_B:
            for B in config_A_B:
                for C in config_C:
                    for D in config_D_E:
                        for E in config_D_E:
                            config = {'A':A,'B':B,'C':C,'D':D,'E':E}
                            self.configs.append(config)

        self.Ad, self.Au = self.make_param((out_features, in_features), f'LoRA_{rank}')
        self.Bd, self.Bu = self.make_param((out_features, in_features), f'LoRA_{rank}')
        self.Cd, self.Cu = self.make_param((in_features, 1), f'LoRA_{rank}')
        self.D = nn.Parameter(torch.zeros(out_features))
        self.E = nn.Parameter(torch.zeros(out_features))
        self.eval_config = None
        nn.init.xavier_uniform_(self.Au)
        nn.init.xavier_uniform_(self.Bu)
        nn.init.xavier_uniform_(self.Cu)

    def prepare_path(self, config: str, Xd: torch.Tensor, Xu: torch.Tensor = None):
        if Xu is not None:
            if 'LoRA' in config:
                rank = int(config.split('_')[1])
                X = torch.matmul(Xd[:,:rank], Xu[:rank, :])
            elif 'vector' in config:
                X = Xd[:,0].unsqueeze(1)
            elif 'constant' in config:
                X = Xd[0,0]
            elif 'none' in config:
                X = torch.zeros(Xd.shape[0], Xu.shape[1]).cuda()
            else:
                raise ValueError
        else:
            if 'vector' in config:
                X = Xd
            elif 'constant' in config:
                X = Xd[0]
            elif 'none' in config:
                X = torch.zeros(1).cuda()
            else:
                raise ValueError
        return X

    def make_param(self, shape: Tuple, config: Optional[str]=None):
        if 'LoRA' in config:
            out_feature = shape[0]
            in_feature = shape[1]
            try:
                rank = int(config.split('_')[1])
            except:
                rank = 4
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape))

    def forward(self, input: torch.Tensor):
        if self.eval_config is not None:
            path_config = self.eval_config
        else:
            path_config = random.choice(self.configs)
        A = self.prepare_path(path_config['A'], self.Ad, self.Au)
        B = self.prepare_path(path_config['B'], self.Bd, self.Bu)
        C = self.prepare_path(path_config['C'], self.Cd, self.Cu)
        D = self.prepare_path(path_config['D'], self.D)
        E = self.prepare_path(path_config['E'], self.E)
        optimal_weight = self.weight + self.weight*A + B
        if torch.is_tensor(self.bias):
            optimal_bias = self.bias + self.bias*D + E
        else:
            optimal_bias = E
        optimal_prompt = torch.matmul(self.weight, C).squeeze()
        return F.linear(input, optimal_weight, optimal_bias+optimal_prompt)

    @staticmethod
    def from_linear(linear_module: nn.Module, rank: int):
        new_linear = SuperScalableLinear(linear_module.in_features, linear_module.out_features, rank)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear

class GLoRAModuleInjection:

    @staticmethod
    def make_scalable(linear_module: nn.Module, rank: int = 4):
        """Make a (linear) layer super scalable.
        :param linear_module: A Linear module
        :return: a suepr linear that can be trained to
        """
        new_linear = SuperScalableLinear.from_linear(linear_module, rank)
        return new_linear