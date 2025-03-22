#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python3 -m pdb -c continue evaluation/eval_freevocab.py
    
#laion2b_s39b_b160k

