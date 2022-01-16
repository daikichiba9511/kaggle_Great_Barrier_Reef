#!/bin/sh

[ ! -d "./yolov5" ] && git clone git@github.com:ultralytics/yolov5.git
cd yolov5
python -m pip install -qr requirements.txt
