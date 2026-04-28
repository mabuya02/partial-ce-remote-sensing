import torchvision.models.segmentation as segmentation
from torchvision.models import ResNet50_Weights


def build_model(num_classes=5, pretrained=True):
    weights = None
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained else None

    model = segmentation.deeplabv3_resnet50(
        weights=weights,
        weights_backbone=weights_backbone,
        num_classes=num_classes
    )
    return model
