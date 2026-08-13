# VisionAD Source Reading Notes

Date: 2026-07-05

Source copied to:
`references/VisionAD`

Repository:
`https://github.com/Qiqigeww/VisionAD`

Paper title:
`Search is All You Need for Few-shot Anomaly Detection`

## What It Uses As Training Material

For the main few-shot test path, VisionAD does not train on abnormal test
samples.

It samples only a few normal support images from each target class:

- MVTec path: `item/train/good`
- selection: `shots` images
- test path: `item/test/*`

The selection is implemented in `MVTecDatasetFS`, which loads only `train/good`
and randomly selects `shots` normal images with a fixed seed.

Important file:
`references/VisionAD/dataset.py`

## Main Training-Free Method

The central lightweight path is `DADE`.

Important files:

- `references/VisionAD/dade_mvtec_test.py`
- `references/VisionAD/models/uad.py`
- `references/VisionAD/utils.py`

The flow is:

1. Load a frozen ViT foundation model, usually DINOv2 register ViT.
2. Pick many intermediate layers, e.g. layers 4 to 18 for DINOv2-L.
3. Extract query patch tokens from all selected layers.
4. Extract support patch tokens from few normal support images.
5. Fuse multi-layer features by averaging the selected layer features.
6. Normalize query/support patch features.
7. For every query patch, search nearest support patch.
8. Anomaly score is nearest-neighbor distance:

```text
cross_distance = 1 - query_patch @ support_patch.T
patch_score = min(cross_distance)
```

Then the patch score is reshaped into an anomaly map.

## Support And Query Augmentation

VisionAD has two augmentation ideas:

1. Support augmentation inside `DADE`:
   - rotate support by 90, 180, 270 degrees
   - vertical flip
   - horizontal flip
   - concatenate all augmented support patches into the support memory

2. Query/support test-time augmentation inside `evaluation_batch`:
   - original prediction
   - flipped query/support prediction, then flip anomaly map back
   - clamped-value prediction
   - sum the three anomaly maps

This is different from our failed generic normal augmentation. VisionAD applies
paired query/support transformations and aggregates predictions, instead of only
blindly expanding the normal gallery.

## Image Score

VisionAD uses the anomaly map for image-level score:

```text
if max_ratio == 0:
    image_score = max(anomaly_map)
else:
    image_score = mean(top max_ratio pixels)
```

Their MVTec script uses:

```text
max_ratio = 0.01
```

So the paper's image score is closer to top-1% anomaly region average, not plain
global average.

## Other Heavier Paths

The repo also contains heavier variants:

- `DADEv2`: trains only limited parameters, mainly the encoder `cls_token`, with
  InfoNCE between query/support normal images.
- `FADE/FADEv3`: cross-attention decoder style models, requiring saved
  checkpoints.

For our current RF few-shot direction, the most relevant part is the
training-free `DADE` path, not FADE.

## Useful Ideas For Our RF Method

1. Use DINOv2 or another stronger vision backbone as an optional visual branch.
   VisionAD relies on DINOv2 register ViT rather than CLIP text semantics.

2. Use many intermediate layers, not just two CLIP ViT layers.
   Their method averages a large set of selected ViT layers.

3. Treat support augmentation and query augmentation symmetrically.
   Our previous normal-only augmentation widened the normal gallery and hurt
   some anomalies. VisionAD transforms query and support together, then fuses
   anomaly maps.

4. Use top-ratio image scoring.
   Their default image score uses top 1% of anomaly map values, which matches our
   observation that max/top-k patch scores are stronger than broad averages.

5. Keep the method training-free for the main line.
   This fits our few-shot RF setting better than adding another trainable head.

## Caution For RF Spectrograms

VisionAD's support rotations/flips are natural for industrial images, but they
are not automatically valid for RF spectrograms:

- vertical flip changes time order
- horizontal flip changes frequency order
- 90-degree rotation swaps time and frequency

For our RF data, these transforms need domain-specific replacements, such as:

- paired time shift for query/support
- very small frequency shift only if physically valid
- amplitude-preserving normalization variants
- no 90-degree rotation unless explicitly justified

