from __future__ import annotations

import torch.nn as nn

from detectors.compact_cnn import build_compact_binary_cnn


def build_vision_backbone(variant: str, *, pretrained: bool, dropout: float):
    if variant == "compact_cnn_v1":
        return build_compact_binary_cnn(dropout=dropout)

    if variant == "efficientnet_b0_ft_v1":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        model.classifier[1] = nn.Linear(in_features, 1)
        return model

    raise ValueError(f"Unsupported vision backbone variant: {variant}")
