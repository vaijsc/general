import os
import yaml
import torch
import argparse
import numpy as np
from munch import Munch
from tqdm import tqdm, trange

# Util
from util2d.util import masks_to_rle
from util2d.maskwise_freevocab import FreeVocab_MaskWise
from util2d.sppwise_freevocab import FreeVocab_SPPWise
from util2d.pointwise_freevocab import FreeVocab_PointWise
from util2d.reproduce_freevocab_pointwise import FreeVocab_Reproduce

from loader3d.scannetpp import INSTANCE_BENCHMARK84_SCANNET_PP
from loader3d.scannet200 import INSTANCE_CAT_SCANNET_200


def rle_encode_gpu_batch(masks):
    """
    Encode RLE (Run-length-encode) from 1D binary mask.
    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1

        runs[1::2] -= runs[::2]

        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles

############################################## FreeVocab SPP-WISE ##############################################
'''
Generate 3D proposals from SAM-2
'''

np.random.seed(0)
torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration FreeVocab")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    if cfg.data.dataset_name == 'scannetpp':
        class_names = INSTANCE_BENCHMARK84_SCANNET_PP
    elif cfg.data.dataset_name == 'scannet200':
        class_names = INSTANCE_CAT_SCANNET_200
    
    # Fondation model loader
    if 'maskwise' in args.config: 
        print("Doing Maskwise")
        model = FreeVocab_MaskWise(cfg, class_names = class_names)
    elif 'sppwise' in args.config: 
        print("Doing SPPtwise")
        model = FreeVocab_SPPWise(cfg, class_names = class_names)
    elif 'pointwise' in args.config:
        print("Doing Pointwise")
        model = FreeVocab_PointWise(cfg, class_names = class_names)
    else:
        print("Doing Reproduce Pointwise")
        model = FreeVocab_Reproduce(cfg, class_names = class_names)

    # Directory Init
    freevocab_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.freevocab_output)
    llm_feature_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.llm_feature)

    os.makedirs(freevocab_path, exist_ok=True)
    os.makedirs(llm_feature_path, exist_ok=True)

    # Proces every scene
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            #####################################
            # Tracker
            done = False
            path = scene_id + ".pth"
            with open("tracker_freevocab.txt", "r") as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    if path in line:
                        done = True
                        break
            if done == True:
                print("existed " + path)
                continue
            # # Write append each line
            with open("tracker_freevocab.txt", "a") as file:
                file.write(path + "\n")
            # # Skip Large Scenes
            if scene_id == '27dd4da69e' or scene_id == '9071e139d9' or scene_id == 'bde1e479ad':
                continue
            if os.path.exists('../nips25/Queryable_FreeVocab/result_pointwise_sub/osm_feature_scannetpp/detic/' + path) == False:
                print('SKIP')
                continue
            #####################################
            # scene_id = 'scene011_00'
            # scene_id = '0d2ee665be'

            #####################################
            print("Process", scene_id)          
            feature = model.refine_freevocab(scene_id, cfg, genfeature = False)  
            print('----------------freevocab Query----------------')
            # Save 3D mask
            proposals3d, confidences, categories = model.get_final_instance(scene_id, cfg)
            cluster_dict = {"ins": rle_encode_gpu_batch(proposals3d), 'conf': confidences, 'name': categories}
            torch.save(cluster_dict, os.path.join(freevocab_path, f"{scene_id}.pth"))            
            # Free memory
            torch.cuda.empty_cache()
