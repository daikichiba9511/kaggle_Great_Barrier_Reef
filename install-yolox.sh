#!/bin/sh

if [ ! -d "./YOLOX" ]; then

git clone https://github.com/Megvii-BaseDetection/YOLOX

cd YOLOX
python -m pip install -U pip && python -m pip install -r requirements.txt
python -m pip install -v -e .

fi

# get weights
# wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
cd -
