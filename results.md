# FracAtlas ResNet18 U-Net Results

This document records the current segmentation experiments and their saved
artifacts. The original notebook cells and original Vanilla U-Net implementation
are frozen. The CLAHE implementation is appended independently at the end of
`FracAtlas_Fracture_Segmentation_Research.ipynb`.

## Experiment state

### Shared architecture

- **Task:** binary fracture-mask segmentation on complete FracAtlas radiographs.
- **Input size:** `3 x 640 x 512`; masks are `1 x 640 x 512`.
- **Encoder:** torchvision ResNet18 initialized with ImageNet pretrained weights.
- **Decoder:** U-Net decoder with skip connections:
  - center: `512 -> 512` `1x1` convolution
  - decoder blocks: `512 -> 256`, `256 -> 128`, `128 -> 64`, `64 -> 32`
  - final upsample: `32 -> 16`, followed by a one-channel `1x1` output head
- **Parameter count:** `14,499,265`.
- **Loss:** `0.5 * BCEWithLogitsLoss + 0.5 * soft Dice loss`.
- **Optimizer:** AdamW, weight decay `0.0001`.
- **Training:** 20 epochs, physical batch size 2, gradient accumulation 2.
- **Learning rates:** decoder `1e-4`; trainable encoder `1e-5`.
- **Augmentation:** training-only horizontal flip and rotation in `[-10, 10]` degrees.
- **Evaluation:** Dice, IoU, precision, recall, HD95, and BCE from raw logits.
- **HD95 convention:** both masks empty = `0`; exactly one mask empty = `inf`;
  distances are reported in original-image pixels.

### CLAHE preprocessing

The CLAHE experiments use the existing split manifests and masks without
modification.

- Input is read as grayscale.
- OpenCV CLAHE configuration: `clipLimit=2.0`, `tileGridSize=(8, 8)`.
- The enhanced grayscale image is repeated into three identical channels.
- Original filenames and spatial dimensions are preserved.
- Training and evaluation normalization uses mean
  `0.18653708696365356` and standard deviation `0.200462207198143`.
- CLAHE evaluation thresholds are selected on validation data only. Both saved
  CLAHE test summaries selected threshold `0.3`.

### Frozen Vanilla U-Net baseline

The original Vanilla U-Net is included as a read-only baseline. Its saved
summary reports 715 annotated images, split into 499 training, 108 validation,
and 108 test images. It uses the original-image preprocessing pipeline,
`3 x 640 x 512` tensors, channels `[16, 32, 64, 128, 256]`, and has
`1,942,577` trainable parameters.

## ResNet18 U-Net comparison

All values below are means over the saved test evaluation summaries.

| Experiment | Dice | IoU | Precision | Recall | HD95 (px) | BCE | Training (s) | Inference (s/image) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original images, frozen encoder | 0.486809 | 0.414414 | 0.619284 | 0.598488 | 70.606647 | 0.019201 | 502.341743 | 0.096239 |
| Original images, two-phase fine-tuning | 0.457925 | 0.349790 | 0.696972 | 0.442681 | 56.022440 | 0.024495 | 558.776956 | 0.094717 |
| CLAHE images, frozen encoder | 0.514331 | 0.433706 | 0.630775 | 0.587602 | 98.275108 | 0.022877 | 433.530461 | 0.097538 |
| CLAHE images, two-phase fine-tuning | 0.467629 | 0.354886 | 0.583666 | 0.511716 | 80.715548 | 0.019132 | 572.385182 | 0.098048 |

The Vanilla U-Net baseline is reported separately because its original summary
does not contain inference-time fields. The supplementary test-metrics JSON
provides BCE and finite HD95 statistics:

| Experiment | Dice | IoU | Precision | Recall | HD95 (px) | BCE | Training (s) | Inference (s/image) | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original images, Vanilla U-Net | 0.235798 | 0.163175 | 0.285955 | 0.283308 | 158.991499* | 0.033471 | 635.731101 | N/A | 1,942,577 |

The Vanilla U-Net best validation Dice was `0.201711` at epoch 18. Its
original summary records peak GPU memory of `1399.28 MB`. The finite Vanilla
HD95 standard deviation is `112.873602` pixels, with `5` infinite HD95 cases;
the aggregate HD95 mean and standard deviation are therefore null under the
saved empty-mask convention. `*` denotes the finite-case HD95 mean.

Each ResNet18 model has `14,499,265` parameters. The displayed values are
loaded from JSON summaries, not recomputed or estimated in this document.

## Validation threshold selection for CLAHE

| Mode | Selected threshold | Validation Dice at selected threshold |
|---|---:|---:|
| CLAHE frozen encoder | 0.50 | 0.362525 |
| CLAHE two-phase fine-tuning | 0.50 | 0.433090 |

The selected threshold is frozen before test evaluation.

## Training modes

### Original-image frozen encoder

The ResNet18 encoder remains frozen for the full training schedule. Decoder
parameters are optimized at `1e-4`.

### Original-image two-phase model

The encoder is frozen for the first two epochs. From epoch 3 onward, encoder
parameters are trainable at `1e-5`, while decoder parameters remain at `1e-4`.

### CLAHE frozen encoder

This is an independently initialized ImageNet-pretrained ResNet18 U-Net using
CLAHE images. The encoder remains frozen for all epochs.

### CLAHE two-phase model

This is an independently initialized ImageNet-pretrained ResNet18 U-Net using
CLAHE images. The encoder is frozen for two epochs and then unfrozen with the
lower `1e-5` encoder learning rate.

## Saved artifact locations

### Original ResNet18 U-Net

- Frozen checkpoint: `outputs/checkpoints/resnet18_unet/frozen_encoder/best.pth`
- Two-phase checkpoint: `outputs/checkpoints/resnet18_unet/two_phase/best.pth`
- Frozen test JSON: `outputs/evaluation/resnet18_unet/frozen_encoder/resnet18_unet_summary.json`
- Two-phase test JSON: `outputs/evaluation/resnet18_unet/two_phase/resnet18_unet_summary.json`
- Histories: `outputs/histories/resnet18_unet/`

### Vanilla U-Net baseline

- Checkpoint: `outputs/checkpoints/vanilla_unet/best.pth`
- Test summary: `outputs/evaluation/vanilla_unet/vanilla_unet_summary.json`
- Supplementary test metrics: `outputs/evaluation/test_metrics.json`
- Per-image metrics: `outputs/evaluation/vanilla_unet/vanilla_unet_per_image_metrics.csv`
- Training history: `outputs/histories/vanilla_unet/vanilla_unet_history.json`

### CLAHE ResNet18 U-Net

- Frozen checkpoint: `outputs/checkpoints/resnet_unet_with_clahe/frozen_encoder/best.pth`
- Two-phase checkpoint: `outputs/checkpoints/resnet_unet_with_clahe/two_phase/best.pth`
- Frozen summary: `outputs/evaluation/resnet_unet_with_clahe/frozen_encoder/summary.json`
- Two-phase summary: `outputs/evaluation/resnet_unet_with_clahe/two_phase/summary.json`
- Frozen history: `outputs/histories/resnet_unet_with_clahe/frozen_encoder/training_history.json`
- Two-phase history: `outputs/histories/resnet_unet_with_clahe/two_phase/training_history.json`
- Per-image metrics: each mode's `evaluation` directory contains
  `per_image_metrics.csv`; validation results are under `validation/`.

### CLAHE preprocessing and reporting

- Enhanced images: `outputs/preprocessing/clahe/images/`
- CLAHE configuration: `outputs/preprocessing/clahe/clahe_config.json`
- Preprocessing statistics: `outputs/preprocessing/clahe/preprocessing_statistics.json`
- Comparison benchmark: `outputs/visualizations/benchmarks/clahe_resnet_comparison.json`
- Displayable benchmark table: `outputs/visualizations/benchmarks/clahe_resnet_comparison.csv`
- Experiment configuration: `outputs/experiment_configs/resnet_unet_with_clahe.json`
  or a versioned `resnet_unet_with_clahe_vN.json` when an existing configuration
  is preserved.

## Source of the reported values

The benchmark display loads:

1. `outputs/evaluation/resnet18_unet/frozen_encoder/resnet18_unet_summary.json`
2. `outputs/evaluation/resnet18_unet/two_phase/resnet18_unet_summary.json`
3. `outputs/evaluation/resnet_unet_with_clahe/frozen_encoder/summary.json`
4. `outputs/evaluation/resnet_unet_with_clahe/two_phase/summary.json`
5. `outputs/evaluation/vanilla_unet/vanilla_unet_summary.json`

Training times are summed from the corresponding saved
`training_history.json` files. No previous model was retrained to produce this
report.
