"""
Model Definitions — EfficientNet-B0 & ResNet-50 for Defect Detection
=====================================================================
Transfer learning models with custom classifier heads for binary
classification (good / defective). Supports layer freezing and
provides Grad-CAM target layer references.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class DefectDetectorEfficientNet(nn.Module):
    """
    EfficientNet-B0 fine-tuned for transistor defect detection.
    
    Architecture:
        EfficientNet-B0 backbone (ImageNet pretrained)
        → AdaptiveAvgPool2d
        → Dropout(0.3)
        → Linear(1280, 256) → ReLU → Dropout(0.2)
        → Linear(256, 2)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Get the number of features from the backbone
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Replace the classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.67),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        """Return the target layer for Grad-CAM (last conv block)."""
        # We target the Conv2d layer directly ([0]) inside the Conv2dNormActivation 
        # to avoid the inplace SiLU activation which breaks pytorch-grad-cam
        return self.backbone.features[-1][0]

    def freeze_backbone(self):
        """Freeze all backbone layers (only classifier is trainable)."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen — only classifier is trainable")

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] All layers unfrozen for fine-tuning")

    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DefectDetectorResNet(nn.Module):
    """
    ResNet-50 fine-tuned for transistor defect detection.
    
    Architecture:
        ResNet-50 backbone (ImageNet pretrained)
        → AdaptiveAvgPool2d
        → Dropout(0.3)
        → Linear(2048, 256) → ReLU → Dropout(0.2)
        → Linear(256, 2)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Get the number of features from the backbone
        in_features = self.backbone.fc.in_features  # 2048

        # Replace the fully connected head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.67),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        """Return the target layer for Grad-CAM (last bottleneck block)."""
        return self.backbone.layer4[-1]

    def freeze_backbone(self):
        """Freeze all backbone layers except the fc head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("[Model] Backbone frozen — only fc head is trainable")

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] All layers unfrozen for fine-tuning")

    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




class DefectDetectorConvNeXt(nn.Module):
    """ConvNeXt-Tiny fine-tuned for transistor defect detection."""
    def __init__(self, num_classes: int = 2, dropout: float = 0.3, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm((in_features,), eps=1e-6, elementwise_affine=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.67),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        return self.backbone.features[-1][-1]

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen - only classifier is trainable")

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All layers unfrozen for fine-tuning")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DefectDetectorEfficientNetV2(nn.Module):
    """EfficientNet-V2-S fine-tuned for transistor defect detection."""
    def __init__(self, num_classes: int = 2, dropout: float = 0.3, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_v2_s(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.67),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        return self.backbone.features[-1]

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen - only classifier is trainable")

    def unfreeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All layers unfrozen for fine-tuning")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_model(
    architecture: str = "efficientnet_b0",
    num_classes: int = 2,
    dropout: float = 0.3,
    hidden_dim: int = 256,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> Tuple[nn.Module, object]:
    """
    Factory function to create a defect detection model.
    
    Args:
        architecture: Model architecture ('efficientnet_b0' or 'resnet50').
        num_classes: Number of output classes.
        dropout: Dropout rate for the classifier.
        hidden_dim: Hidden dimension in the classifier.
        pretrained: Use ImageNet pretrained weights.
        freeze_backbone: Freeze backbone for initial training.
    
    Returns:
        Tuple of (model, gradcam_target_layer).
    """
    architecture = architecture.lower().strip()

    if architecture == "efficientnet_b0":
        model = DefectDetectorEfficientNet(
            num_classes=num_classes, dropout=dropout,
            hidden_dim=hidden_dim, pretrained=pretrained
        )
    elif architecture == "resnet50":
        model = DefectDetectorResNet(
            num_classes=num_classes, dropout=dropout,
            hidden_dim=hidden_dim, pretrained=pretrained
        )
    elif architecture == "convnext_tiny":
        model = DefectDetectorConvNeXt(
            num_classes=num_classes, dropout=dropout,
            hidden_dim=hidden_dim, pretrained=pretrained
        )
    elif architecture == "efficientnet_v2_s":
        model = DefectDetectorEfficientNetV2(
            num_classes=num_classes, dropout=dropout,
            hidden_dim=hidden_dim, pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'efficientnet_b0', 'resnet50', 'convnext_tiny', or 'efficientnet_v2_s'.")

    if freeze_backbone:
        model.freeze_backbone()

    gradcam_target_layer = model.get_gradcam_target_layer()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.get_trainable_params()
    print(f"\n[Model] {architecture}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {total_params - trainable_params:,}")

    return model, gradcam_target_layer
