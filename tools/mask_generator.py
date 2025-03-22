import os
import yaml
import torch
import argparse
import numpy as np
from munch import Munch
from tqdm import tqdm, trange

# Util
from util2d.util import masks_to_rle
from util2d.segment_anything_v2 import SAM_L2
from util2d.open3dis_sam2 import Open3DIS_SAM_L2

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

############################################## Mask Generator ##############################################
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


    # Fondation model loader
    if cfg.segmenter2d.model == 'SAM-2':
        model = SAM_L2(cfg)
    elif cfg.segmenter2d.model == 'Open3DIS_SAM-2':
        model = Open3DIS_SAM_L2(cfg)

    # Directory Init
    save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    mask2d_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)

    os.makedirs(save_dir_cluster, exist_ok=True)
    os.makedirs(mask2d_path, exist_ok=True)

    # Proces every scene
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            #####################################
            # Tracker
            done = False
            path = scene_id + ".pth"
            with open("tracker_2d.txt", "r") as file:
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
            with open("tracker_2d.txt", "a") as file:
                file.write(path + "\n")
            # # Skip Large Scenes
            if scene_id == '27dd4da69e' or scene_id == '9071e139d9' or scene_id == 'bde1e479ad':
                continue
            #####################################
            # scene_id = '0d2ee665be'
            # scene_id = 'scene0011_00'
            print("Process", scene_id)
            proposals3d, mask2d_bank, id_bank, obj_bank = model.generate3dproposal(
                scene_id,
                cfg=cfg,
            )

            # Save 3D mask
            cluster_dict = {"ins": rle_encode_gpu_batch(proposals3d), 'mask2d_bank': mask2d_bank, 'id_bank': id_bank, 'obj_bank': obj_bank}
            torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))            
            # Free memory
            torch.cuda.empty_cache()
