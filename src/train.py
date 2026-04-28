import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from src.dataset import AerialSegmentationDataset, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from src.model import build_model
from src.losses import partial_cross_entropy_loss
from src.utils import sample_point_labels
from src.metrics import mean_iou, per_class_iou, pixel_accuracy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tile_level_indices(samples, val_tiles):
    train_indices = []
    val_indices = []

    for idx, sample in enumerate(samples):
        tile_name = Path(sample["image_path"]).parts[-3]
        if tile_name in val_tiles:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    return train_indices, val_indices


def append_history(history_path, row):
    existing_rows = []
    fieldnames = list(row.keys())

    if history_path.exists():
        with history_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    existing_rows.append(row)

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow(existing_row)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dataset = AerialSegmentationDataset(args.csv_path, image_size=args.image_size)
    train_source = AerialSegmentationDataset(
        args.csv_path,
        image_size=args.image_size,
        augment=args.augment,
    )
    val_source = AerialSegmentationDataset(args.csv_path, image_size=args.image_size)

    if args.split == "tile":
        train_indices, val_indices = tile_level_indices(base_dataset.samples, set(args.val_tiles))
        if not train_indices or not val_indices:
            raise ValueError(
                "Tile split produced an empty train or validation set. "
                "Check --val-tiles against the tile names in data/splits.csv."
            )
        train_dataset = Subset(train_source, train_indices)
        val_dataset = Subset(val_source, val_indices)
    else:
        train_size = int(0.8 * len(base_dataset))
        val_size = len(base_dataset) - train_size

        train_indices, val_indices = random_split(
            range(len(base_dataset)),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        train_dataset = Subset(train_source, list(train_indices))
        val_dataset = Subset(val_source, list(val_indices))

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

    model = build_model(
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrained,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    if args.supervision == "full":
        run_name = f"{args.split}_full"
    else:
        run_name = f"{args.split}_partial_ratio_{args.point_ratio}"
    best_miou = -1.0

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, masks in progress:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]

            if args.supervision == "partial":
                sparse_targets = []
                point_masks = []

                for mask in masks:
                    sparse_target, point_mask = sample_point_labels(
                        mask.cpu(),
                        ratio=args.point_ratio,
                        ignore_index=IGNORE_INDEX,
                    )
                    sparse_targets.append(sparse_target)
                    point_masks.append(point_mask)

                sparse_targets = torch.stack(sparse_targets).to(device)
                point_masks = torch.stack(point_masks).to(device)

                loss = partial_cross_entropy_loss(
                    outputs,
                    sparse_targets,
                    point_masks,
                    ignore_index=IGNORE_INDEX,
                )
            else:
                loss = F.cross_entropy(outputs, masks, ignore_index=IGNORE_INDEX)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        model.eval()
        total_acc = 0.0
        total_miou = 0.0
        total_class_ious = [0.0 for _ in range(NUM_CLASSES)]
        total_class_counts = [0 for _ in range(NUM_CLASSES)]

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)["out"]

                acc = pixel_accuracy(outputs, masks, ignore_index=IGNORE_INDEX)
                miou = mean_iou(outputs, masks, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
                class_ious = per_class_iou(
                    outputs,
                    masks,
                    num_classes=NUM_CLASSES,
                    ignore_index=IGNORE_INDEX,
                )

                total_acc += acc.item()
                total_miou += miou.item()
                for class_idx, class_iou in enumerate(class_ious):
                    if class_iou is not None:
                        total_class_ious[class_idx] += class_iou.item()
                        total_class_counts[class_idx] += 1

        avg_acc = total_acc / len(val_loader)
        avg_miou = total_miou / len(val_loader)
        avg_class_ious = [
            total / count if count > 0 else 0.0
            for total, count in zip(total_class_ious, total_class_counts)
        ]

        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg_loss:.4f}, "
            f"val_acc={avg_acc:.4f}, "
            f"val_miou={avg_miou:.4f}"
        )

        history_row = {
            "supervision": args.supervision,
            "point_ratio": args.point_ratio if args.supervision == "partial" else "",
            "epoch": epoch + 1,
            "epochs": args.epochs,
            "loss": f"{avg_loss:.6f}",
            "val_acc": f"{avg_acc:.6f}",
            "val_miou": f"{avg_miou:.6f}",
            "split": args.split,
            "pretrained": str(not args.no_pretrained),
            "augment": str(args.augment),
            "seed": args.seed,
        }
        for class_name, class_iou in zip(CLASS_NAMES, avg_class_ious):
            history_row[f"iou_{class_name.lower()}"] = f"{class_iou:.6f}"

        append_history(history_path, history_row)

        if avg_miou > best_miou:
            best_miou = avg_miou
            best_model_path = output_dir / f"best_model_{run_name}.pth"
            torch.save(model.state_dict(), best_model_path)

    model_path = output_dir / f"model_{run_name}.pth"
    torch.save(model.state_dict(), model_path)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv-path", type=str, default="data/splits.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--point-ratio", type=float, default=0.01)
    parser.add_argument("--supervision", choices=["partial", "full"], default="partial")
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["random", "tile"], default="random")
    parser.add_argument("--val-tiles", nargs="+", default=["Tile 8"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")

    args = parser.parse_args()

    train(args)
