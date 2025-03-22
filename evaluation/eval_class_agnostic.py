import os

import numpy as np
import torch
from loader3d.scannet200 import INSTANCE_CAT_SCANNET_200
from loader3d.scannetpp import SEMANTIC_CAT_SCANNET_PP, INSTANCE_BENCHMARK84_SCANNET_PP, SEMANTIC_INSTANCE_CAT_SCANNET_PP # ScannetPP
from scannetv2_inst_eval import ScanNetEval
from tqdm import tqdm
import argparse
import yaml
from munch import Munch

def rle_decode(rle):
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except:
        s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    parser.add_argument("--type",type=str,required = True,help="[2D, 3D, 2D_3D]") # raw 3DIS

    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    eval_type= args.type
    
    if cfg.data.dataset_name  == 'scannet200':
        scan_eval = ScanNetEval(class_labels=INSTANCE_CAT_SCANNET_200, use_label = False, dataset_name = 'scannet200')
        pcl_path = cfg.data.gt_pth # groundtruth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass
    if cfg.data.dataset_name  == 'scannetpp': # 
        # eval on 1554instance classes, inputting sem+ins set
        scan_eval = ScanNetEval(class_labels=SEMANTIC_INSTANCE_CAT_SCANNET_PP, use_label = False, dataset_name = 'scannetpp')
        pcl_path = cfg.data.gt_pth # groundtruth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass
    if cfg.data.dataset_name  == 'scannetpp_benchmark': # 
        # eval on 84 top instance classes
        scan_eval = ScanNetEval(class_labels=INSTANCE_BENCHMARK84_SCANNET_PP, use_label = False, dataset_name = 'scannetpp_benchmark')
        pcl_path = cfg.data.gt_pth # groundtruth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass


    # parent_folder = "/root/3dllm/minhlnh"
    # exp_folder = "freevocab_exp_scannetpp/version_sam2_512_aware_track_5views_30weighted/mask2d_lifted"
    # exp_folder = "freevocab_exp_scannetpp/version_sam2_512_aware_track_5views_30weighted_test_new_maskopt3/mask2d_lifted"
    # data_path = os.path.join(parent_folder, exp_folder)
    # scenes = sorted([s for s in os.listdir(data_path) if s.endswith(".pth")])

    gtsem = []
    gtinst = []
    res = [] #ScannetV2
    
    # data_path = os.path.join(parent_folder, exp_folder)
    # scenes = ['scene0011_00.pth', 'scene0011_01.pth', 'scene0015_00.pth', 'scene0019_00.pth', 'scene0019_01.pth', 'scene0025_00.pth', 'scene0025_01.pth', 'scene0025_02.pth', 'scene0030_00.pth', 'scene0030_01.pth', 'scene0030_02.pth', 'scene0046_00.pth', 'scene0046_01.pth', 'scene0046_02.pth', 'scene0050_00.pth', 'scene0050_01.pth', 'scene0050_02.pth', 'scene0063_00.pth', 'scene0064_00.pth', 'scene0064_01.pth', 'scene0077_00.pth', 'scene0077_01.pth', 'scene0081_00.pth', 'scene0081_01.pth', 'scene0081_02.pth', 'scene0084_00.pth', 'scene0084_01.pth', 'scene0084_02.pth', 'scene0086_00.pth', 'scene0086_01.pth', 'scene0086_02.pth', 'scene0088_00.pth','scene0088_01.pth', 'scene0088_02.pth', 'scene0088_03.pth', 'scene0095_00.pth', 'scene0095_01.pth', 'scene0100_00.pth', 'scene0100_01.pth', 'scene0100_02.pth', 'scene0131_00.pth', 'scene0131_01.pth', 'scene0131_02.pth', 'scene0139_00.pth', 'scene0144_00.pth', 'scene0144_01.pth', 'scene0146_00.pth', 'scene0146_01.pth', 'scene0146_02.pth', 'scene0149_00.pth', 'scene0153_00.pth', 'scene0153_01.pth', 'scene0164_00.pth', 'scene0164_01.pth', 'scene0164_02.pth', 'scene0164_03.pth', 'scene0169_00.pth', 'scene0169_01.pth', 'scene0187_00.pth', 'scene0187_01.pth', 'scene0193_00.pth', 'scene0193_01.pth', 'scene0196_00.pth', 'scene0203_00.pth','scene0203_01.pth', 'scene0203_02.pth', 'scene0207_00.pth', 'scene0207_01.pth', 'scene0207_02.pth', 'scene0208_00.pth', 'scene0217_00.pth', 'scene0221_00.pth', 'scene0221_01.pth', 'scene0222_00.pth', 'scene0222_01.pth', 'scene0231_00.pth']
    ### Scene SubSet
    # scenes = ['09c1414f1b.pth', '0d2ee665be.pth', '13c3e046d7.pth', '1ada7a0617.pth', '21d970d8de.pth', '25f3b7a318.pth', '286b55a2bf.pth']
    # data_path = '../freevocab_exp_scannetpp/version_sam2_512_aware_track_5views_10weighted/mask2d_lifted'
    

    data_path = "/root/3dllm/minhlnh/freevocab_exp_scannetpp/version_dp_maximum_score_0.6_n_spp_div4/mask2d_lifted"

    data_path3D = "/root/3dllm/minhlnh/FreeVocab-3DIS/data/Scannet200/Scannet200_3D/class_ag_res_200_isbnetfull"

    scenes = os.listdir(data_path)

    # scenes = ["scene0011_00.pth"]
    # scenes = [scenes[0]]
    for scene in tqdm(scenes):
        if scene == '27dd4da69e.pth' or scene == '9071e139d9.pth' or scene == 'bde1e479ad.pth':
            continue
        scene_path = os.path.join(data_path, scene)
        scene_path3D = os.path.join(data_path3D, scene)
        
        # scene_path1 = os.path.join("../freevocab_exp_scannetpp/version_sam2_512_no_post_process/mask2d_lifted/", scene)
        try: # skipping heavy scenes
            pred_mask = torch.load(scene_path)
            # pred_mask3D = torch.load(scene_path3D)
            # pred_mask1 = torch.load(scene_path1)
        except:
            print('SKIP: ', scene)
            continue

        gt_path = os.path.join(pcl_path, scene)
        loader = torch.load(gt_path)
        sem_gt, inst_gt = loader[2], loader[3]
        gtsem.append(np.array(sem_gt).astype(np.int32))
        gtinst.append(np.array(inst_gt).astype(np.int32))
        
        masks = pred_mask['ins']        
        
        n_mask = len(masks)
        tmp = []
        
        # masks3D = pred_mask3D['ins']
        # for mask in masks3D:
        #     conf = 1.0
        #     scene_id = scene.replace('.pth', '')
        #     tmp.append({"scan_id": scene_id, "label_id": 0, "conf": conf, "pred_mask": mask}) # class-agnostic evaluation

        for ind in range(n_mask):
            if isinstance(masks[ind], dict):
                mask = rle_decode(masks[ind])
            else:
                try:
                    mask = (masks[ind] == 1).numpy().astype(np.uint8)
                except:
                    mask = (masks[ind] == 1).astype(np.uint8)

            # conf = score[ind] #
            conf = 1.0

            scene_id = scene.replace('.pth', '')
            tmp.append({"scan_id": scene_id, "label_id": 0, "conf": conf, "pred_mask": mask}) # class-agnostic evaluation
        res.append(tmp)

    scan_eval.evaluate(res, gtsem, gtinst)
