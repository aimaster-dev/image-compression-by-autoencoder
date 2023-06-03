# Image compression using Auto-Encoder neural network with ResNet-34 bottleneck blocks

## Introduction

This project is a simple implementation of auto-encoder neural network for image compression. The auto-encoder neural
network is trained on the ImageNet dataset. The trained model is then used to compress and decompress the images. The
compressed images are stored in a file and can be decompressed later.

## Training

```shell
python train.py \
  --root [path to dataset] \
  --epochs [number of epochs] \
  --batch_size [batch size] \
  --lr [learning rate] \
  --save_path [path to save model] \
  --device [torch device to train on] \
  --vgg_alpha [alpha for vgg loss] \
  --latent_dim [latent dimension] \
  --quantize_levels [number of quantization levels]

```

## Compression

```shell
python compress.py \
  --root [path to images] \
  --model_path [path to model] \
  --device [torch device to train on] \
  --quantize_levels [number of quantization levels] \
  --compressed_path [path to save compressed image]
```

## Decompression

```shell
python decompress.py \
  --compressed_path [path to compressed image] \
  --model_path [path to model] \
  --device [torch device to train on] \
  --quantize_levels [number of quantization levels] \
  --decompressed_path [path to save decompressed image]
```