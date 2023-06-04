# Compress all files in a assets/images folder

QB=$1
RESNET=$2
DEVICE=$3

for file in $(ls -1 assets/images); do
  python compress.py \
    --image=assets/images/$file \
    --output=assets/compressed/${file%.*}.bin \
    --models-dir=models \
    --qb=$QB \
    --resnet-model=$RESNET \
    --device=$DEVICE

done
