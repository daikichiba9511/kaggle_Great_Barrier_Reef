#!/bin/sh

[ ! -d "./mmdetection" ] && git clone https://github.com/open-mmlab/mmdetection.git

pip install -q openmim

mim install mmdet