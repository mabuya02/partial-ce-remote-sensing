import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation


def build_model(num_classes=6):
    model = segmentation.deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes
    )
    return model