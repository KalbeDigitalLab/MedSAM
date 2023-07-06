# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import List, Optional, Tuple, Type

from .common import MLPBlock, AdapterMLPBlock, AdditionAdapterMLPBlock, LoRALayer


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class AdapterTwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        block: TwoWayAttentionBlock,
        mlp_dim: int = 64,
        scale: float = 0.1,

    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          mlp_ratio (int): Size of mlp hidden dim to embedding dim.
          scale (int): mlp residual adapter scaling factor

        """
        super().__init__()

        # lets freeze first
        for parameter in block.parameters():
            parameter.requires_grad = False

        embedding_dim = block.mlp.lin1.in_features

        self.mlp_adapter = AdapterMLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim)
        self.down_sample_queries = nn.Linear(embedding_dim, mlp_dim=mlp_dim)
        self.token_to_image_adapter = AdditionAdapterMLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim)
        self.image_to_token_adapter = AdapterMLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim)
        self.scale = scale
        self.block = block

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.block.skip_first_layer_pe:
            queries = self.block.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.block.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.block.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.block.cross_attn_token_to_image(q=q, k=k, v=keys)

        downsampled_queries = self.down_sample_queries(queries)
        queries = queries + attn_out

        queries = self.token_to_image_adapter(queries, downsampled_queries)
        queries = queries + keys

        # MLP block
        queries = self.block.mlp(self.block.norm2(queries)) + self.scale * self.mlp_adapter(queries)
        queries = self.block.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.block.cross_attn_image_to_token(q=k, k=q, v=queries)
        attn_out = attn_out + keys
        attn_out = self.image_to_token_adapter(attn_out)
        keys = keys + attn_out
        keys = self.block.norm4(keys)

        return queries, keys


class LoRATwoWayTransformer(nn.Module):
    def __init__(
        self,
        transformer: TwoWayTransformer,
        r: int = 4,
        lora_layer: Optional[List] = None,
    ) -> None:
        super(LoRATwoWayTransformer, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(transformer.layers)))

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in transformer.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(transformer.layers):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            # self attn
            attn_w_q_linear = blk.self_attn.q_proj
            attn_w_v_linear = blk.self_attn.v_proj
            base_dim = attn_w_q_linear.in_features
            attn_w_a_linear_q = nn.Linear(base_dim, r, bias=False)
            attn_w_b_linear_q = nn.Linear(r, base_dim, bias=False)
            attn_w_a_linear_v = nn.Linear(base_dim, r, bias=False)
            attn_w_b_linear_v = nn.Linear(r, base_dim, bias=False)
            self.w_As.append(attn_w_a_linear_q)
            self.w_Bs.append(attn_w_b_linear_q)
            self.w_As.append(attn_w_a_linear_v)
            self.w_Bs.append(attn_w_b_linear_v)
            blk.self_attn.q_proj = LoRALayer(attn_w_q_linear, attn_w_a_linear_q, attn_w_b_linear_q)
            blk.self_attn.v_proj = LoRALayer(attn_w_v_linear, attn_w_a_linear_v, attn_w_b_linear_v)

            # token to image
            token_w_q_linear = blk.cross_attn_token_to_image.q_proj
            token_w_v_linear = blk.cross_attn_token_to_image.v_proj
            base_dim = token_w_q_linear.in_features
            token_w_a_linear_q = nn.Linear(base_dim, r, bias=False)
            token_w_b_linear_q = nn.Linear(r, token_w_q_linear.out_features, bias=False)
            token_w_a_linear_v = nn.Linear(base_dim, r, bias=False)
            token_w_b_linear_v = nn.Linear(r, token_w_v_linear.out_features, bias=False)
            self.w_As.append(token_w_a_linear_q)
            self.w_Bs.append(token_w_b_linear_q)
            self.w_As.append(token_w_a_linear_v)
            self.w_Bs.append(token_w_b_linear_v)
            blk.cross_attn_token_to_image.q_proj = LoRALayer(token_w_q_linear, token_w_a_linear_q, token_w_b_linear_q)
            blk.cross_attn_token_to_image.v_proj = LoRALayer(token_w_v_linear, token_w_a_linear_v, token_w_b_linear_v)

            # image to token
            img_w_q_linear = blk.cross_attn_image_to_token.q_proj
            img_w_v_linear = blk.cross_attn_image_to_token.v_proj
            base_dim = img_w_q_linear.in_features
            img_w_a_linear_q = nn.Linear(base_dim, r, bias=False)
            img_w_b_linear_q = nn.Linear(r, img_w_q_linear.out_features, bias=False)
            img_w_a_linear_v = nn.Linear(base_dim, r, bias=False)
            img_w_b_linear_v = nn.Linear(r, img_w_v_linear.out_features, bias=False)
            self.w_As.append(img_w_a_linear_q)
            self.w_Bs.append(img_w_b_linear_q)
            self.w_As.append(img_w_a_linear_v)
            self.w_Bs.append(img_w_b_linear_v)
            blk.cross_attn_image_to_token.q_proj = LoRALayer(img_w_q_linear, img_w_a_linear_q, img_w_b_linear_q)
            blk.cross_attn_image_to_token.v_proj = LoRALayer(img_w_v_linear, img_w_a_linear_v, img_w_b_linear_v)

        self.reset_parameters()
        self.transformer = transformer

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer(x)


class AdapterTwoWayTransformer(nn.Module):
    def __init__(
        self,
        transformer: TwoWayTransformer,
        scale: float = 0.1,
        mlp_dim: int = 64,
    ) -> None:
        super(AdapterTwoWayTransformer, self).__init__()

        # lets freeze first
        for param in transformer.parameters():
            param.requires_grad = False

        adapter_layers = nn.ModuleList()

        # Here, we do the surgery
        if isinstance(transformer, LoRATwoWayTransformer):
            for _, blk in enumerate(transformer.transformer.layers):
                adapter_layers.append(AdapterTwoWayAttentionBlock(blk, mlp_dim, scale))
            transformer.transformer.layers = adapter_layers

        elif isinstance(transformer, TwoWayTransformer):
            for _, blk in enumerate(transformer.layers):
                adapter_layers.append(AdapterTwoWayAttentionBlock(blk, mlp_dim, scale))
            transformer.layers = adapter_layers

        self.transformer = transformer

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer(x)


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
