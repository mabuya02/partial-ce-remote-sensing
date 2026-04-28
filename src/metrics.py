import torch


def pixel_accuracy(logits, targets, ignore_index=5):
    preds = torch.argmax(logits, dim=1)

    valid = targets != ignore_index
    correct = (preds == targets) & valid

    return correct.sum().float() / valid.sum().clamp(min=1)


def mean_iou(logits, targets, num_classes=6, ignore_index=5):
    preds = torch.argmax(logits, dim=1)

    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        pred_cls = preds == cls
        target_cls = targets == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union > 0:
            ious.append(intersection / union)

    if len(ious) == 0:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(ious).mean()