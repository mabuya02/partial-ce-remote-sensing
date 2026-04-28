# Technical Report: Partial Cross Entropy for Point-Supervised Remote Sensing Segmentation

## Method

The task is semantic segmentation with limited point-level supervision. A standard segmentation model predicts a class for every pixel, but the training objective only receives labels at sparse point locations.

I implemented partial cross entropy loss. For each image, a binary point mask marks the sampled labeled pixels. Cross entropy is computed per pixel, multiplied by the point mask, and averaged over only the labeled points:

```text
L = sum(CE(logits, target) * point_mask) / sum(point_mask)
```

The full masks are used to simulate point annotations and to evaluate validation performance. Unknown and unlabeled pixels are mapped to `ignore_index=255`, so they do not contribute to training or validation metrics.

The segmentation network is DeepLabV3 with a ResNet-50 backbone. The improved implementation supports a pretrained ImageNet backbone, optional flip augmentation, fixed random seeds, random or tile-level validation splits, metric logging, and qualitative visualization panels.

## Dataset

The experiments use the Semantic Segmentation of Aerial Imagery dataset, available on Kaggle:

https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery

This dataset contains high-resolution aerial images together with pixel-level segmentation masks. The images represent urban and semi-urban scenes captured from above, so they include structures and surfaces that are important in remote-sensing analysis such as buildings, roads, land, vegetation, and water. In the original dataset, the masks are stored as color-coded RGB images, where each color corresponds to one semantic class.

According to the Kaggle description, the dataset was published by Humans in the Loop as part of a joint project with the Mohammed Bin Rashid Space Center (MBRSC) in Dubai, UAE. The imagery comes from Dubai aerial and satellite scenes, and the annotations were produced as pixel-wise semantic segmentation masks. Kaggle also notes that the segmentation work was carried out by trainees from the Roia Foundation in Syria.

For this project, the full masks were used in two ways. First, they were used to simulate sparse point supervision by sampling only a small number of labeled pixels from each image. Second, they were used as ground truth during validation so that the point-supervised model could be compared fairly against a full-supervision baseline. The classes used for training and evaluation are:

| Class id | Class |
| --- | --- |
| 0 | Building |
| 1 | Land |
| 2 | Road |
| 3 | Vegetation |
| 4 | Water |

The original mask annotations use the following color codes:

| Class | Color |
| --- | --- |
| Building | `#3C1098` |
| Land (unpaved area) | `#8429F6` |
| Road | `#6EC1E4` |
| Vegetation | `#FEDD3A` |
| Water | `#E2A929` |
| Unlabeled | `#9B9B9B` |

The original unlabeled class and any unknown mask colors are mapped to `ignore_index=255`, so they do not contribute to the loss or evaluation metrics.

## Experiment 1: Effect of Point Label Density

### Purpose and Hypothesis

The purpose is to test whether increasing the number of point labels improves segmentation performance. The hypothesis is that higher point density should improve mIoU because the network receives more spatial supervision per class.

### Process

I randomly sampled point labels from the full masks using three point ratios:

| Setting | Description |
| --- | --- |
| `0.01` | 1% of pixels sampled per class |
| `0.05` | 5% of pixels sampled per class |
| `0.10` | 10% of pixels sampled per class |

Each model was trained for 20 epochs with batch size 2, ImageNet-pretrained ResNet-50 backbone weights, random flip augmentation, and seed `42`. Validation used pixel accuracy and mean IoU.

### Results

The table reports the best validation mIoU achieved during training and the final epoch result.

| Point ratio | Best epoch | Best val accuracy | Best val mIoU | Final val accuracy | Final val mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 18 | 0.6856 | 0.3733 | 0.6784 | 0.3573 |
| 0.05 | 18 | 0.6936 | 0.3882 | 0.6802 | 0.3612 |
| 0.10 | 18 | 0.6916 | 0.3722 | 0.6921 | 0.3714 |

The best result was achieved with 5% point labels, reaching `0.3882` validation mIoU. The 1%, 5%, and 10% settings all improved rapidly during the first few epochs and converged to a similar range by epoch 18. This suggests that the model benefits from sparse point supervision, but simply increasing point density beyond 5% did not provide a clear additional gain under this setup.

## Analysis

The corrected experiment demonstrates that partial CE can train a segmentation model from sparse point labels. Compared with the initial 3-epoch prototype, the best mIoU improved from about `0.1586` to `0.3882`. The main reasons for the improvement are longer training, pretrained backbone weights, augmentation, and correct ignored-pixel handling.

The hypothesis was partially supported. More supervision helped compared with the initial weak setup, but 10% point labels did not outperform 5%. This may be because the model is limited more by dataset size, class imbalance, and spatial generalization than by point density once enough points are available.

Remaining limitations:

- Only one random seed was used.
- The dataset is small, so metrics can be noisy.
- The full-supervision baseline was trained for the same 20 epochs, but longer training may separate the upper bound more clearly.

## Experiment 2: Tile-Level Validation

### Purpose

The random split experiment may overestimate generalization because nearby remote-sensing image patches can be spatially correlated. To test a stricter setup, I held out all samples from `Tile 8` for validation and trained on the remaining tiles using the best point-density setting from Experiment 1.

### Result

| Split | Point ratio | Best epoch | Best val accuracy | Best val mIoU | Final val accuracy | Final val mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random image split | 0.05 | 18 | 0.6936 | 0.3882 | 0.6802 | 0.3612 |
| Tile 8 holdout | 0.05 | 20 | 0.6411 | 0.3280 | 0.6411 | 0.3280 |

The tile-level score is lower than the random image split score, which is expected because the validation tile is spatially unseen. This result is more conservative and gives a better estimate of cross-tile generalization.

Using the standalone evaluator with ignored pixels masked in both predictions and targets, the saved Tile 8 checkpoint gives `0.3342` mIoU. Per-class IoU is:

| Class | IoU |
| --- | ---: |
| Building | 0.3539 |
| Land | 0.6564 |
| Road | 0.2085 |
| Vegetation | 0.1877 |
| Water | 0.2647 |

This confirms the qualitative pattern: large land regions are learned best, while roads and vegetation are weaker.

## Experiment 3: Full-Supervision Baseline

### Purpose

The full-mask cross entropy baseline provides an upper-bound reference using the same network, optimizer, augmentation, and splits, but with every labeled pixel used in the loss. This shows how close partial point supervision gets to standard full supervision in this setup.

### Result

| Split | Supervision | Best epoch | Val accuracy | Val mIoU |
| --- | --- | ---: | ---: | ---: |
| Random image split | 5% point labels | 18 | 0.6936 | 0.3882 |
| Random image split | Full masks | 20 | 0.6907 | 0.3926 |
| Tile 8 holdout | 5% point labels | 20 | 0.6411 | 0.3342 |
| Tile 8 holdout | Full masks | 19 | 0.6191 | 0.3311 |

The random split full-supervision baseline is slightly higher than the point-supervised result, but the gap is small (`0.3926` vs `0.3882` mIoU). On the stricter tile holdout, the 5% point-label model is slightly higher than the full-supervision baseline (`0.3342` vs `0.3311` mIoU). This should not be interpreted as point supervision being generally superior; it more likely reflects limited training time, small validation size, augmentation randomness, and model variance. The important result is that partial CE with 5% point labels reaches nearly the same performance as full-mask supervision under the same training budget.

Per-class IoU for the full-supervision tile checkpoint is:

| Class | IoU |
| --- | ---: |
| Building | 0.3299 |
| Land | 0.6116 |
| Road | 0.2078 |
| Vegetation | 0.2151 |
| Water | 0.2909 |

## Qualitative Assessment

The visualization panels show that the model learns broad semantic regions such as land, water, and vegetation. The best examples are visible in the selected visualization folders, especially samples `50`, `52`, and `53` from the `0.10` checkpoint and sample `50` from the `0.05` checkpoint.

The main qualitative weakness is fine structure. Thin roads, narrow boundaries, and small building regions are often smoothed away or merged into the dominant surrounding class. This matches the moderate mIoU values: the model captures coarse regions but still struggles with small objects and precise boundaries.

The default visualization command writes fixed filenames, so repeated runs can overwrite earlier panels. For comparison, separate output folders were generated for each point density:

```text
outputs/visualizations_ratio_0.01/
outputs/visualizations_ratio_0.05/
outputs/visualizations_ratio_0.10/
outputs/visualizations_selected_ratio_0.01/
outputs/visualizations_selected_ratio_0.05/
outputs/visualizations_selected_ratio_0.10/
```

## Improvements Added

The implementation has been improved in the following ways:

- Unknown and unlabeled pixels are now ignored with `ignore_index=255`.
- The model predicts only five valid semantic classes.
- Training supports pretrained ResNet-50 backbone weights.
- Training supports random seeds and writes `outputs/history.csv`.
- Training supports tile-level validation splits.
- Training supports simple flip augmentation.
- A visualization script generates side-by-side panels of image, ground truth, point labels, and prediction.
- `README.md` and `requirements.txt` document reproducible usage.

## Reproducibility Commands

The main experiment was run with:

```bash
python -m src.train --epochs 20 --batch-size 2 --point-ratio 0.01 --augment
python -m src.train --epochs 20 --batch-size 2 --point-ratio 0.05 --augment
python -m src.train --epochs 20 --batch-size 2 --point-ratio 0.10 --augment
```

Qualitative outputs can be generated with:

```bash
python -m src.visualize --checkpoint outputs/best_model_random_ratio_0.01.pth --point-ratio 0.01
python -m src.visualize --checkpoint outputs/best_model_tile_ratio_0.05.pth --point-ratio 0.05
python -m src.visualize --checkpoint outputs/best_model_random_ratio_0.1.pth --point-ratio 0.10
```

The tile-level validation experiment was run with:

```bash
python -m src.train --epochs 20 --batch-size 2 --point-ratio 0.05 --split tile --val-tiles "Tile 8" --augment
```

The full-supervision upper-bound baselines were run with:

```bash
python -m src.train --epochs 20 --batch-size 2 --supervision full --augment
python -m src.train --epochs 20 --batch-size 2 --supervision full --split tile --val-tiles "Tile 8" --augment
```

Per-class IoU can be computed with:

```bash
python -m src.evaluate --checkpoint outputs/best_model_tile_ratio_0.05.pth --split tile-val --val-tiles "Tile 8" --output-csv outputs/evaluation.csv
python -m src.evaluate --checkpoint outputs/best_model_tile_full.pth --split tile-val --val-tiles "Tile 8" --output-csv outputs/evaluation.csv
```

Metric-selected qualitative examples can be generated with:

```bash
python -m src.visualize --checkpoint outputs/best_model_tile_ratio_0.05.pth --point-ratio 0.05 --output-dir outputs/visualizations_tile_best --select-by-miou best
python -m src.visualize --checkpoint outputs/best_model_tile_ratio_0.05.pth --point-ratio 0.05 --output-dir outputs/visualizations_tile_worst --select-by-miou worst
```

## Conclusion

The project satisfies the core requirement: partial cross entropy loss is implemented and connected to a remote-sensing segmentation network using simulated point labels. The best random-split point-supervised experiment achieved `0.3882` validation mIoU with 5% point labels, very close to the full-supervision baseline at `0.3926`. On the stricter Tile 8 holdout, the point-supervised model achieved `0.3342` mIoU and the full-supervision baseline achieved `0.3311`. The results show that sparse point supervision is viable for this remote-sensing segmentation task and can approach full-mask supervision under the same training budget.
