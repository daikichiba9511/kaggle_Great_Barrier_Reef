#!/bin/sh

echo ' ####### start to train ######## '

FILE_NAME=exp011.py

if [ $1 = "debug" ]; then
        python exp/${FILE_NAME} \
                --debug \
                --train_fold 0
fi

if [ $1 = "all" ]; then
        python exp/${FILE_NAME} \
                --train_fold 1 2 3 4
fi

if [ $1 = "val" ]; then
        python exp/${FILE_NAME} \
                --train_fold 0
fi
