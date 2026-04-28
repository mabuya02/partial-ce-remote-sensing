# Point-Supervised Remote Sensing Segmentation

This project implements partial cross entropy loss for semantic segmentation with sparse point labels. Full segmentation masks are used only to simulate point supervision and evaluate predictions; the training loss is computed only at sampled point locations.

## Dataset

The implementation uses the Semantic Segmentation Dataset from Dubai satellite imagery. The data is indexed into `data/splits.csv` with paired image and mask paths.

Mask classes:

| Class id | Class |
| --- | --- |
| 0 | Building |
| 1 | Land |
| 2 | Road |
| 3 | Vegetation |
| 4 | Water |

Unknown and unlabeled pixels are assigned `ignore_index=255` and excluded from training loss and validation metrics.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Prepare the CSV index after placing the raw dataset under `data/raw/Semantic segmentation dataset`:

```bash
python -m src.prepare_data
```

## Training

Train with 1% simulated point labels:

```bash
python -m src.train --epochs 20 --batch-size 2 --point-ratio 0.01 --augment
```

Run the main point-density experiment:

```bash
python -m src.train --epochs 20 --batch-size 2 --point-ratio 0.01 --augment
python -m src.train --epochs 20 --batch-size 2 --point-ratio 0.05 --augment
python -m src.train --epochs 20 --batch-size 2 --point-ratio 0.10 --augment
```

Run the full-supervision upper-bound baseline:

```bash
python -m src.train --epochs 20 --batch-size 2 --supervision full --augment
python -m src.train --epochs 20 --batch-size 2 --supervision full --split tile --val-tiles "Tile 8" --augment
```

For a stricter spatial validation split, hold out a full tile:

```bash
python -m src.train --epochs 20 --batch-size 2 --point-ratio 0.05 --split tile --val-tiles "Tile 8" --augment
```

Training writes model checkpoints and `outputs/history.csv`.
Checkpoints are split-aware for new runs, for example `outputs/best_model_random_ratio_0.1.pth` or `outputs/best_model_tile_ratio_0.05.pth`.

## Evaluation

Evaluate a checkpoint and print pixel accuracy, mIoU, and per-class IoU:

```bash
python -m src.evaluate --checkpoint outputs/best_model_tile_ratio_0.05.pth --split tile-val --val-tiles "Tile 8"
```

Save evaluation rows to CSV:

```bash
python -m src.evaluate --checkpoint outputs/best_model_tile_ratio_0.05.pth --split tile-val --val-tiles "Tile 8" --output-csv outputs/evaluation.csv
```

## Visualizations

Create qualitative panels for the report:

```bash
python -m src.visualize --checkpoint outputs/best_model_tile_ratio_0.05.pth --point-ratio 0.05 --output-dir outputs/visualizations_tile_ratio_0.05
```

Generate selected qualitative examples:

```bash
python -m src.visualize --checkpoint outputs/best_model_random_ratio_0.1.pth --point-ratio 0.10 --output-dir outputs/visualizations_selected_ratio_0.10 --indices 50 52 53 71
```

Automatically choose examples by checkpoint mIoU:

```bash
python -m src.visualize --checkpoint outputs/best_model_tile_ratio_0.05.pth --point-ratio 0.05 --output-dir outputs/visualizations_tile_best --select-by-miou best
python -m src.visualize --checkpoint outputs/best_model_tile_ratio_0.05.pth --point-ratio 0.05 --output-dir outputs/visualizations_tile_worst --select-by-miou worst
```

Each panel contains the input image, ground truth, sampled point labels, and model prediction.

## Results

Corrected 20-epoch results with pretrained backbone and flip augmentation:

| Point ratio | Best epoch | Best val accuracy | Best val mIoU | Final val mIoU |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | 18 | 0.6856 | 0.3733 | 0.3573 |
| 0.05 | 18 | 0.6936 | 0.3882 | 0.3612 |
| 0.10 | 18 | 0.6916 | 0.3722 | 0.3714 |

The strongest run was `point-ratio=0.05`, reaching `0.3882` validation mIoU.

Tile-level validation with `Tile 8` held out is stricter and reached:

| Split | Point ratio | Best epoch | Best val accuracy | Best val mIoU | Final val mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tile 8 holdout | 0.05 | 20 | 0.6411 | 0.3280 | 0.3280 |

The standalone evaluator reports `0.3342` mIoU for the saved Tile 8 checkpoint after masking ignored pixels in both predictions and targets.

Full-supervision baseline:

| Split | Supervision | Best epoch | Val accuracy | Val mIoU |
| --- | --- | ---: | ---: | ---: |
| Random image split | Full masks | 20 | 0.6907 | 0.3926 |
| Tile 8 holdout | Full masks | 19 | 0.6191 | 0.3311 |

The 5% point-supervised model is close to the full-mask baseline under the same 20-epoch training budget.
