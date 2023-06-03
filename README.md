# Image compression using variational auto-encoder

This project is a simple implementation of auto-encoder neural network for image compression.
The auto-encoder neural network is trained on the ImageNet dataset. The trained model is then used to compress and
decompress the images.

## Model architecture

Model represents a variational auto-encoder with residual blocks and skip connections.

* Encoder: _ResNet-18 architecture with fully connected layers_
* Decoder: _ResNet-18 architecture with transposed convolution layers_
* Loss: _VGG loss + MSE loss_

## Download pretrained models

Models were trained on ImageNet dataset subset (20000 images).

Here are the links to download the pretrained models:
_B = number of quantization levels_

* [B=2, resnet18]()
* [B=4, resnet18]()

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
  --root [path to images] \
  --resnet-model [resnet model architecture] \
  --epochs [number of epochs] \
  --batch_size [batch size] \
  --lr [learning rate] \
  --device [torch device to train on] \
  --vgg_alpha [weight of VGG loss] \
  --quantize_levels [number of quantization levels] \
  --save_results_every [save results every n epochs] \
  --save_models_dir [path to save models]
```

## Results

### Compression

| Original image size | Compressed image size (B=2, L=128) | Compressed image size (B=4, L=256) |
|---------------------|------------------------------------|------------------------------------|
