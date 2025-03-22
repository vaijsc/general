import os
import yaml
import torch
import argparse
import numpy as np
from munch import Munch
from tqdm import tqdm, trange

# Util
from util2d.util import masks_to_rle
# from util2d.direct_openvocab import OpenVocab_Direct
# from util2d.sppwise_openvocab import OpenVocab_SPPWise
# from util2d.pointwise_openvocab import OpenVocab_PointWise
from util2d.reproduce_openvocab_pointwise import OpenVocab_Reproduce_PointWise

# from util2d.reproduce_openvocab import OpenVocab_SPPWise
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

############################################## OpenVocab SPP-WISE ##############################################
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
    model = OpenVocab_Reproduce_PointWise(cfg, class_names = class_names)

    # Directory Init
    openvocab_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.openvocab_output)
    clipfeature_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clip_feature)

    os.makedirs(openvocab_path, exist_ok=True)
    os.makedirs(clipfeature_path, exist_ok=True)

    # Proces every scene
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            #####################################
            # Tracker
            done = False
            path = scene_id + ".pth"    
            with open("tracker_openvocab.txt", "r") as file:
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
            with open("tracker_openvocab.txt", "a") as file:
                file.write(path + "\n")
            # # Skip Large Scenes
            if scene_id == '27dd4da69e' or scene_id == '9071e139d9' or scene_id == 'bde1e479ad':
                continue
            #####################################
            # scene_id = '0d2ee665be'
            # scene_id = 'scene0011_00'
            print("Process", scene_id)
            print('----------------OpenVocab Query----------------')
            # Save 3D mask
            feature_bank = model.refine_openvocab(
                scene_id,
                cfg=cfg,
                genfeature = False # Whether or not run feature generator
            )
            # feature_dict = {"feat": feature_bank}
            # torch.save(feature_dict, os.path.join(clipfeature_path, f"{scene_id}.pth"))  
            proposals3d, class_id, categories = model.get_final_instance(scene_id, cfg)
            cluster_dict = {"ins": rle_encode_gpu_batch(proposals3d), 'class': class_id, 'name': categories}
            torch.save(cluster_dict, os.path.join(openvocab_path, f"{scene_id}.pth"))            
            # Free memory
            torch.cuda.empty_cache()


