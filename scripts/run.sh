#!/bin/bash
sh scripts/mask_generator.sh configs/scannet200.yaml
sh scripts/mask_freevocab.sh configs/scannet200_sppwise_freevocab.yaml
