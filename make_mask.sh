#!/bin/bash

echo "The first argument received is: $1"

source /home/jess/anaconda3/etc/profile.d/conda.sh

conda run --no-capture-output -n sam32 python3 /home/jess/sam3/make_masks.py "$1"

