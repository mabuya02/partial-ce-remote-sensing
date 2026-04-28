import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import AerialSegmentationDataset
from src.model import build_model
from src.losses import partial_cross_entropy_loss
from src.utils import sample_point_labels
from src.metrics import pixel_accuracy, mean_iou


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AerialSegmentationDataset("data/splits.csv")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = build_model(num_classes=6).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, masks in progress:
            images = images.to(device)
            masks = masks.to(device)

            sparse_targets = []
            point_masks = []

            for mask in masks:
                sparse_target, point_mask = sample_point_labels(
                    mask.cpu(),
                    ratio=args.point_ratio
                )
                sparse_targets.append(sparse_target)
                point_masks.append(point_mask)

            sparse_targets = torch.stack(sparse_targets).to(device)
            point_masks = torch.stack(point_masks).to(device)

            outputs = model(images)["out"]

            loss = partial_cross_entropy_loss(
                outputs,
                sparse_targets,
                point_masks
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        model.eval()
        total_acc = 0.0
        total_miou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)["out"]

                acc = pixel_accuracy(outputs, masks)
                miou = mean_iou(outputs, masks)

                total_acc += acc.item()
                total_miou += miou.item()

        avg_acc = total_acc / len(val_loader)
        avg_miou = total_miou / len(val_loader)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg_loss:.4f}, "
            f"val_acc={avg_acc:.4f}, "
            f"val_miou={avg_miou:.4f}"
        )

    model_path = output_dir / f"model_ratio_{args.point_ratio}.pth"
    torch.save(model.state_dict(), model_path)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--point-ratio", type=float, default=0.01)

    args = parser.parse_args()

    train(args)