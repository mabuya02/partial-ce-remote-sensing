import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.dataset import AerialSegmentationDataset, IGNORE_INDEX, NUM_CLASSES
from src.metrics import mean_iou
from src.model import build_model
from src.utils import sample_point_labels


CLASS_COLORS = {
    0: (60, 16, 152),
    1: (132, 41, 246),
    2: (110, 193, 228),
    3: (254, 221, 58),
    4: (226, 169, 41),
    IGNORE_INDEX: (40, 40, 40),
}


def tensor_image_to_pil(image):
    array = image.permute(1, 2, 0).numpy()
    array = np.clip(array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def mask_to_pil(mask):
    mask_np = mask.cpu().numpy()
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        rgb[mask_np == class_id] = color

    return Image.fromarray(rgb)


def sparse_points_to_pil(mask, point_mask):
    image = Image.new("RGB", (mask.shape[1], mask.shape[0]), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    ys, xs = torch.where(point_mask > 0)

    for y, x in zip(ys.tolist(), xs.tolist()):
        color = CLASS_COLORS.get(int(mask[y, x]), (255, 255, 255))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    return image


def labeled_panel(title, image):
    title_h = 24
    panel = Image.new("RGB", (image.width, image.height + title_h), (255, 255, 255))
    panel.paste(image, (0, title_h))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 5), title, fill=(0, 0, 0))
    return panel


def make_panel(image, target, point_target, point_mask, pred):
    panels = [
        labeled_panel("Image", tensor_image_to_pil(image)),
        labeled_panel("Ground truth", mask_to_pil(target)),
        labeled_panel("Point labels", sparse_points_to_pil(point_target, point_mask)),
        labeled_panel("Prediction", mask_to_pil(pred)),
    ]

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    output = Image.new("RGB", (width, height), (255, 255, 255))

    x = 0
    for panel in panels:
        output.paste(panel, (x, 0))
        x += panel.width

    return output


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AerialSegmentationDataset(args.csv_path, image_size=args.image_size)
    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.select_by_miou:
        scored_indices = []
        with torch.no_grad():
            for idx in range(len(dataset)):
                image, target = dataset[idx]
                logits = model(image.unsqueeze(0).to(device))["out"]
                score = mean_iou(
                    logits,
                    target.unsqueeze(0).to(device),
                    num_classes=NUM_CLASSES,
                    ignore_index=IGNORE_INDEX,
                )
                scored_indices.append((float(score), idx))

        scored_indices.sort()
        if args.select_by_miou == "worst":
            indices = [idx for _, idx in scored_indices[:args.num_samples]]
        elif args.select_by_miou == "best":
            indices = [idx for _, idx in scored_indices[-args.num_samples:]]
        else:
            middle = len(scored_indices) // 2
            half = args.num_samples // 2
            start = max(0, middle - half)
            indices = [idx for _, idx in scored_indices[start:start + args.num_samples]]
    else:
        indices = args.indices if args.indices else range(min(args.num_samples, len(dataset)))

    for idx in indices:
        image, target = dataset[idx]
        point_target, point_mask = sample_point_labels(
            target,
            ratio=args.point_ratio,
            ignore_index=IGNORE_INDEX,
        )

        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))["out"]
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu()

        panel = make_panel(image, target, point_target, point_mask, pred)
        panel.save(output_dir / f"sample_{idx:02d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/model_ratio_0.01.pth")
    parser.add_argument("--csv-path", type=str, default="data/splits.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations")
    parser.add_argument("--point-ratio", type=float, default=0.01)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--select-by-miou", choices=["best", "median", "worst"])
    parser.add_argument("--image-size", type=int, nargs=2, default=(256, 256))

    main(parser.parse_args())
