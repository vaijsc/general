import os
import yaml
import torch
import argparse
import numpy as np
from munch import Munch
from tqdm import tqdm, trange

# Util
from util2d.util import masks_to_rle
from util2d.prompt_segment_anything_v2 import Prompt_SAM_L2

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
    model = Prompt_SAM_L2(cfg)

    # Directory Init
    save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    mask2d_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)

    os.makedirs(save_dir_cluster, exist_ok=True)
    os.makedirs(mask2d_path, exist_ok=True)

    # Proces every scene
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            ## Tracker
            # done = False
            # path = scene_id + ".pth"
            # with open("tracker_prompt.txt", "r") as file:
            #     lines = file.readlines()
            #     lines = [line.strip() for line in lines]
            #     for line in lines:
            #         if path in line:
            #             done = True
            #             break
            # if done == True:
            #     print("existed " + path)
            #     continue
            # # Write append each line
            # with open("tracker_prompt.txt", "a") as file:
            #     file.write(path + "\n")
            #####################################
            scene_id = '21d970d8de'
            print("Process", scene_id)
            proposals3d, inter, uni = model.generate3dproposal(
                scene_id,
                cfg=cfg,
                promptclick = cfg.proposal3d.prompt_click
            )
            # Save 3D mask
            cluster_dict = {"ins": rle_encode_gpu_batch(proposals3d), "intersect": inter.cpu(), "union": uni.cpu()}
            torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))            
            print('inter:', (inter/uni).mean().item())
            # Free memory
            torch.cuda.empty_cache()
        
