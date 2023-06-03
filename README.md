# Image compression using Auto-Encoder neural network with ResNet-34 bottleneck blocks

## Introduction

This project is a simple implementation of auto-encoder neural network for image compression. The auto-encoder neural
network is trained on the ImageNet dataset. The trained model is then used to compress and decompress the images. The
compressed images are stored in a file and can be decompressed later.

## Download pretrained models

Here are the links to download the pretrained models:

_B = number of quantization levels_, _L = latent dimension_

* [B=2, L=128]()
* [B=4, L=256]()

Put downloaded models in `models` directory.


## Quick example:
```shell
# Compress the `baboon` image from assets/images directory

python compress.py \
  --image=assets/images/baboon.png \
  --models_dir=models \
  --latent_dim=128 \
  --device=cuda \
  --quantize_levels=2 \
  --compressed_path=compressed.bin
  
# Decompress the compressed image
python decompress.py \
  --compressed_path=compressed.bin \
  --models_dir=models \
  --latent_dim=128 \
  --device=cuda \
  --quantize_levels=2 \
  --decompressed_path=decompressed.png
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

## Training from scratch

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
