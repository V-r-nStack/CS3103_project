# Model Architectures

The project uses seven binary fracture-segmentation models. All models receive
`3 x 640 x 512` images and produce one-channel `640 x 512` mask logits.
The original-image models use the grayscale radiograph repeated across three
channels. CLAHE models apply OpenCV CLAHE (`clipLimit=2.0`,
`tileGridSize=(8, 8)`) before channel repetition. All ResNet18 encoders use
ImageNet-pretrained weights.

## Model summary

| Model | Input | Encoder | Decoder / head | Encoder training |
|---|---|---|---|---|
| Vanilla U-Net | Original | Four convolutional encoder stages | Symmetric U-Net decoder | Trainable |
| ResNet18 U-Net, frozen | Original | ResNet18 | U-Net decoder with skips | Frozen |
| ResNet18 U-Net, two-phase | Original | ResNet18 | U-Net decoder with skips | Frozen for epochs 1–2; trainable from epoch 3 |
| ResNet18 U-Net, frozen + CLAHE | CLAHE | ResNet18 | U-Net decoder with skips | Frozen |
| ResNet18 U-Net, two-phase + CLAHE | CLAHE | ResNet18 | U-Net decoder with skips | Frozen for epochs 1–2; trainable from epoch 3 |
| DeepLabV3+ ResNet18 | Original | ResNet18 | ASPP + low-level-feature decoder | Frozen |
| DeepLabV3+ ResNet18 + CLAHE | CLAHE | ResNet18 | ASPP + low-level-feature decoder | Frozen |

## Architecture details

### Vanilla U-Net

- Encoder channels: `[16, 32, 64, 128]`, followed by a `256`-channel
  bottleneck.
- Each stage uses two `3 x 3` convolutions, batch normalization, and ReLU.
- `2 x 2` max pooling downsamples the encoder features.
- The decoder uses `2 x 2` transposed convolutions, concatenates mirrored
  encoder skip features, and applies the same two-convolution blocks.
- A `1 x 1` convolution produces the one-channel output.
- Parameters: **1,942,577**.

### ResNet18 U-Net (four variants)

- The ResNet18 stem and layers 1–4 provide the encoder features and skip
  connections.
- A `1 x 1` center convolution maps `512 -> 512` channels.
- Decoder blocks map `512 -> 256 -> 128 -> 64 -> 32`, using upsampling and
  skip concatenation.
- A final upsampling maps `32 -> 16`, followed by a one-channel `1 x 1`
  output head.
- Parameters: **14,499,265** per variant.
- The frozen models keep the encoder fixed for all 20 epochs.
- The two-phase models keep the encoder fixed for the first two epochs, then
  train it from epoch 3 at learning rate `1e-5`; decoder learning rate is
  `1e-4`.

The four variants differ only in input preprocessing (original or CLAHE) and
encoder training schedule (frozen or two-phase).

### DeepLabV3+ ResNet18 (two variants)

- The ResNet18 stem and layers 1–4 form the frozen encoder.
- A custom ASPP module processes the final `512`-channel feature map using
  parallel `1 x 1`, dilated `3 x 3` convolutions with rates `6, 12, 18`, and
  global average pooling.
- ASPP outputs are projected to `256` channels.
- The low-level layer-1 feature (`64` channels) is projected to `48` channels.
- The high-level feature is bilinearly upsampled to the low-level resolution,
  concatenated (`256 + 48 = 304` channels), and processed by two `3 x 3`
  convolutional blocks with `256` channels.
- A `1 x 1` classifier produces the mask logit, which is bilinearly upsampled
  to `640 x 512`.
- Parameters: **16,602,529** per variant.
- Both variants use the same architecture; only the input preprocessing
  differs (original or CLAHE).
