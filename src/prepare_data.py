from pathlib import Path
import csv

RAW_ROOT = Path("data/raw/Semantic segmentation dataset")
OUTPUT_CSV = Path("data/splits.csv")

pairs = []

for tile_dir in sorted(RAW_ROOT.glob("Tile *")):
    image_dir = tile_dir / "images"
    mask_dir = tile_dir / "masks"

    for image_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / image_path.with_suffix(".png").name

        if mask_path.exists():
            pairs.append((str(image_path), str(mask_path)))

print(f"Found {len(pairs)} image-mask pairs")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_CSV.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "mask_path"])
    writer.writerows(pairs)

print(f"Saved index to {OUTPUT_CSV}")