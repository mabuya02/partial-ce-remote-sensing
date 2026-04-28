import torch


def pixel_accuracy(logits, targets, ignore_index=255):
    preds = torch.argmax(logits, dim=1)

    valid = targets != ignore_index
    correct = (preds == targets) & valid

    return correct.sum().float() / valid.sum().clamp(min=1)


def mean_iou(logits, targets, num_classes=5, ignore_index=255):
    ious = per_class_iou(logits, targets, num_classes=num_classes, ignore_index=ignore_index)
    valid_ious = [iou for iou in ious if iou is not None]

    if len(valid_ious) == 0:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(valid_ious).mean()


def per_class_iou(logits, targets, num_classes=5, ignore_index=255):
    preds = torch.argmax(logits, dim=1)
    valid = targets != ignore_index

    ious = []

    for cls in range(num_classes):
        pred_cls = (preds == cls) & valid
        target_cls = (targets == cls) & valid

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(None)

    return ious
