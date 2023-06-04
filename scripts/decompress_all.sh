# Decompress all files in a assets/compressed folder

QB=$1
RESNET=$2
DEVICE=$3

for file in $(ls -1 assets/compressed/B=$QB); do
  python decompress.py \
    --file=assets/compressed/B=$QB/$file \
    --output=assets/decompressed/B=$QB/${file%.*}.png \
    --models-dir=models \
    --qb=$QB \
    --resnet-model=$RESNET \
    --device=$DEVICE

done