import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from src.dataset import AerialSegmentationDataset, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from src.metrics import mean_iou, per_class_iou, pixel_accuracy
from src.model import build_model
from src.train import tile_level_indices


def build_subset(dataset, args):
    if args.split == "all":
        return dataset

    if args.split in {"random-train", "random-val"}:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_indices, val_indices = random_split(
            range(len(dataset)),
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        indices = val_indices if args.split == "random-val" else train_indices
        return Subset(dataset, list(indices))

    train_indices, val_indices = tile_level_indices(dataset.samples, set(args.val_tiles))
    indices = val_indices if args.split == "tile-val" else train_indices
    return Subset(dataset, indices)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AerialSegmentationDataset(args.csv_path, image_size=args.image_size)
    subset = build_subset(dataset, args)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    total_acc = 0.0
    total_miou = 0.0
    total_class_ious = [0.0 for _ in range(NUM_CLASSES)]
    total_class_counts = [0 for _ in range(NUM_CLASSES)]

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)["out"]

            total_acc += pixel_accuracy(outputs, masks, ignore_index=IGNORE_INDEX).item()
            total_miou += mean_iou(
                outputs,
                masks,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
            ).item()

            class_ious = per_class_iou(
                outputs,
                masks,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
            )
            for class_idx, class_iou in enumerate(class_ious):
                if class_iou is not None:
                    total_class_ious[class_idx] += class_iou.item()
                    total_class_counts[class_idx] += 1

    row = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "val_tiles": " ".join(args.val_tiles),
        "pixel_accuracy": total_acc / len(loader),
        "mean_iou": total_miou / len(loader),
    }
    for class_name, total, count in zip(CLASS_NAMES, total_class_ious, total_class_counts):
        row[f"iou_{class_name.lower()}"] = total / count if count else 0.0

    for key, value in row.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        exists = output_path.exists()
        with output_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv-path", type=str, default="data/splits.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument(
        "--split",
        choices=["all", "random-train", "random-val", "tile-train", "tile-val"],
        default="all",
    )
    parser.add_argument("--val-tiles", nargs="+", default=["Tile 8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str)

    evaluate(parser.parse_args())
