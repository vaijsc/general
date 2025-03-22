#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python3 evaluation/eval_openvocab.py
    
#laion2b_s39b_b160k

