import torch
import torch.nn.functional as F


def partial_cross_entropy_loss(logits, targets, point_mask):
    """
    Partial cross entropy loss for point-supervised segmentation.

    Args:
        logits: Tensor of shape [B, C, H, W]
        targets: Tensor of shape [B, H, W]
        point_mask: Tensor of shape [B, H, W], where 1 = labeled pixel, 0 = ignored pixel

    Returns:
        Scalar loss computed only on labeled pixels.
    """
    pixel_loss = F.cross_entropy(logits, targets, reduction="none")

    masked_loss = pixel_loss * point_mask

    num_labeled_pixels = point_mask.sum().clamp(min=1.0)

    return masked_loss.sum() / num_labeled_pixels