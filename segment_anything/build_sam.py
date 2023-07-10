# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
from typing import List, Optional
import urllib.request
import torch
import copy

from .modeling import (
    ImageEncoderViT,
    LoRAImageEncoderViT,
    AdapterImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
    LoRATwoWayTransformer,
    AdapterTwoWayTransformer,
)


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def apply_encoder_modification(
    sam_model: Sam,
    enable_lora_attn: bool = False,
    enable_adapter_mlp: bool = False,
    adapter_scale: float = 0.1,
    adapter_mlp_dim: int = 64,
    lora_rank: int = 4,
    lora_layer: Optional[List] = None,
    ) -> Sam:
    """Apply Encoder Modification.

    Available modifications:
    1. Adapter from MedSAM-Adapter: arxiv.org/abs/2304.12620
    2. LoRA from Low Rank Adaptation: arxiv.org/abs/2106.09685

    Adapter method will add serial/parallel layers w.r.t attention modules or mlp blocks.
    Lora method will modify attention modules.

    Parameters
    ----------
    sam_model : Sam
        SAM Model
    enable_lora_attn : bool, optional
        Enable LoRA on Attention Modules, by default False
    enable_adapter_mlp : bool, optional
        Enable Adapter on MLP Modules, by default False
    adapter_scale : float, optional
        Value to scale Adapter MLP output, by default 0.1
    adapter_mlp_dim : int, optional
        Size of mlp hidden dim to embedding dim in Adapter block, by default 64
    lora_rank : int, optional
        LoRA rank, by default 4
    lora_layer : Optional[List], optional
        Apply LoRA to selected layers, by default None

    Returns
    -------
    Sam
        Modified sam encoder model.
    """
    if enable_lora_attn and isinstance(sam_model.image_encoder, ImageEncoderViT):
        sam_model.image_encoder = LoRAImageEncoderViT(sam_model.image_encoder, lora_rank, lora_layer)
    if enable_adapter_mlp and isinstance(sam_model.image_encoder, (ImageEncoderViT, LoRAImageEncoderViT)):
        sam_model.image_encoder = AdapterImageEncoderViT(sam_model.image_encoder, adapter_scale, adapter_mlp_dim)
    return sam_model


def apply_decoder_modification(
    sam_model: Sam,
    enable_lora_attn: bool = False,
    enable_adapter_mlp: bool = False,
    adapter_scale: float = 0.1,
    adapter_mlp_dim: int = 64,
    lora_rank: int = 4,
    lora_layer: Optional[List] = None,
    ) -> Sam:
    """Apply Decoder Modification.

    Available modifications:
    1. Adapter from MedSAM-Adapter: arxiv.org/abs/2304.12620
    2. LoRA from Low Rank Adaptation: arxiv.org/abs/2106.09685

    Adapter method will add serial/parallel layers w.r.t attention modules or mlp blocks.
    Lora method will modify attention modules.

    Parameters
    ----------
    sam_model : Sam
        SAM Model
    enable_lora_attn : bool, optional
        Enable LoRA on Attention Modules, by default False
    enable_adapter_mlp : bool, optional
        Enable Adapter on MLP Modules, by default False
    adapter_scale : float, optional
        Value to scale Adapter MLP output, by default 0.1
    adapter_mlp_dim : int, optional
        Size of mlp hidden dim to embedding dim in Adapter block, by default 64
    lora_rank : int, optional
        LoRA rank, by default 4
    lora_layer : Optional[List], optional
        Apply LoRA to selected layers, by default None

    Returns
    -------
    Sam
        Modified sam decoder model.
    """
    if enable_lora_attn and isinstance(sam_model.mask_decoder.transformer, TwoWayTransformer):
        sam_model.mask_decoder.transformer = LoRATwoWayTransformer(sam_model.mask_decoder.transformer, lora_rank, lora_layer)
    if enable_adapter_mlp and isinstance(sam_model.mask_decoder.transformer, (TwoWayTransformer, LoRATwoWayTransformer)):
        sam_model.mask_decoder.transformer = AdapterTwoWayTransformer(sam_model.mask_decoder.transformer, adapter_scale, adapter_mlp_dim)
    return sam_model


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-H checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-L checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")


    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam
