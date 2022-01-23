#!/bin/sh

echo ' ####### start to train ######## '

if [ $1 = "debug" ]; then
        python exp/exp004.py \
                --debug \
                --train_fold 0
fi

if [ $1 = "all" ]; then
        python exp/exp004.py \
                --train_fold 1 2 3 4
fi

if [ $1 = "val" ]; then
        python exp/exp004.py \
                --train_fold 0
fi