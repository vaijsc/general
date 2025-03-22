#!/bin/bash

dataset_cfg=${1:-'configs/scannetpp.yaml'}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./segmenter2d/segment-anything-2:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python3 -m pdb -c continue tools/mask_openvocab.py --config $dataset_cfg
# python /root/minhlnh/sd_utils.py