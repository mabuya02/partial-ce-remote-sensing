import torch


def sample_point_labels(mask, ratio=0.01, ignore_index=5):
    """
    Randomly sample sparse point labels from a full segmentation mask.

    Args:
        mask: Tensor [H, W]
        ratio: Fraction of pixels to sample per class
        ignore_index: Class index to ignore, e.g. unlabeled class

    Returns:
        sparse_target: Tensor [H, W]
        point_mask: Tensor [H, W], 1 where labeled, 0 elsewhere
    """
    point_mask = torch.zeros_like(mask, dtype=torch.float32)
    sparse_target = mask.clone()

    classes = torch.unique(mask)

    for cls in classes:
        cls_value = int(cls.item())

        if cls_value == ignore_index:
            continue

        positions = (mask == cls_value).nonzero(as_tuple=False)

        if positions.numel() == 0:
            continue

        num_points = max(1, int(len(positions) * ratio))

        selected_indices = torch.randperm(len(positions))[:num_points]
        selected_positions = positions[selected_indices]

        point_mask[selected_positions[:, 0], selected_positions[:, 1]] = 1.0

    return sparse_target, point_mask