#!/bin/bash

dataset_cfg=${1:-'configs/scannetpp.yaml'}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./segmenter2d/segment-anything-2:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python3 tools/mask_freevocab.py --config $dataset_cfg

#  -m pdb -c continue