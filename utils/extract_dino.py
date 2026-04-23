# dino_feature_extractor - 与 4DGS-SLAM 一致
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import utils.dinov2 as dinov2


class DinoFeatureExtractor(nn.Module):
    """
    输入:  image 形状 B x 3 x H x W, 值范围 [0, 1]
    输出: 特征图 B x C x H' x W' (H'/W' 约等于 H/W 除以 patch_size, 一般为 /14)
    """

    img_norm_mean: Tensor
    img_norm_std: Tensor

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        assert hasattr(dinov2, backbone_name), f"dinov2 中没有模型 {backbone_name}"
        self.backbone = getattr(dinov2, backbone_name)(pretrained=pretrained)

        self.patch_size = self.backbone.patch_size
        self.embed_dim = self.backbone.embed_dim

        img_norm_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        img_norm_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.register_buffer("img_norm_mean", img_norm_mean / 255.0)
        self.register_buffer("img_norm_std", img_norm_std / 255.0)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _get_pad_1d(self, size: int) -> Tuple[int, int]:
        new_size = math.ceil(size / self.patch_size) * self.patch_size
        pad_size = new_size - size
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        return pad_left, pad_right

    def _pad_to_patch_size(self, x: Tensor):
        _, _, H, W = x.shape
        pad_left, pad_right = self._get_pad_1d(W)
        pad_top, pad_bottom = self._get_pad_1d(H)
        pads = (pad_left, pad_right, pad_top, pad_bottom)
        x = F.pad(x, pads)
        return x, pads

    def _unpad_features(self, feat: Tensor, pads):
        pad_left, pad_right, pad_top, pad_bottom = pads
        ps = self.patch_size

        Hf, Wf = feat.shape[-2:]
        top_idx = pad_top // ps
        left_idx = pad_left // ps
        bottom_idx = Hf - pad_bottom // ps
        right_idx = Wf - pad_right // ps

        return feat[:, :, top_idx:bottom_idx, left_idx:right_idx]

    def forward_N5(self, image: Tensor) -> Tensor:
        assert image.dim() == 4 and image.size(1) == 3, "输入必须是 B x 3 x H x W"

        x = (image - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        x, pads = self._pad_to_patch_size(x)

        with torch.no_grad():
            feats = self.backbone.get_intermediate_layers(
                x,
                n=[self.backbone.num_heads - 1],
                reshape=True,
            )[-1]

        feats = self._unpad_features(feats, pads)
        return feats

    def forward(self, image: Tensor) -> Tensor:
        assert image.dim() == 4 and image.size(1) == 3, "输入必须是 B x 3 x H x W"

        x = (image - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        x, pads = self._pad_to_patch_size(x)

        with torch.no_grad():
            feats_tuple = self.backbone.get_intermediate_layers(
                x,
                n=4,
                reshape=True,
                return_class_token=False,
            )
            feats = torch.stack(feats_tuple, dim=0).mean(dim=0)

        feats = self._unpad_features(feats, pads)
        return feats
