import os

import numpy as np
import torch
from loader3d.scannet200 import INSTANCE_CAT_SCANNET_200
from loader3d.scannetpp import INSTANCE_BENCHMARK84_SCANNET_PP, SEMANTIC_INSTANCE_BENCHMARK84_SCANNET_PP
from scannetv2_inst_eval import ScanNetEval
from tqdm import tqdm

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

# data_path = "../freevocab_exp_scannet200/version_dp/openvocab_results"
# data_path = "/root/3dllm/minhlnh/freevocab_exp_scannetpp/version_dp_top10/openvocab_results"
# data_path = "/root/3dllm/minhlnh/freevocab_exp_scannet200/version_dp/openvocab_results"

data_path = "/root/3dllm/minhlnh/freevocab_exp_scannet200/version_dp_maximum_score_0.6_n_spp_div4/openvocab_results"

data_path = "/root/3dllm/minhlnh/freevocab_exp_scannetpp/version_dp_maximum_score_0.6_n_spp_div4/openvocab_results_masknew"



##NOTE: Scannet++
scan_eval = ScanNetEval(class_labels=SEMANTIC_INSTANCE_BENCHMARK84_SCANNET_PP, dataset_name = 'scannetpp_benchmark')
pcl_path = "./data/Scannetpp/Scannetpp_3D/val/groundtruth_benchmark_instance"

##NOTE: ScanNet200
# scan_eval = ScanNetEval(class_labels=INSTANCE_CAT_SCANNET_200, dataset_name = 'scannet200')
# pcl_path = "./data/Scannet200/Scannet200_3D/val/groundtruth"


if __name__ == "__main__":
    scenes = sorted([s for s in os.listdir(data_path) if s.endswith(".pth")])
    gtsem = []
    gtinst = []
    res = []
    # scenes = ['scene0011_00.pth']
    # scenes = scenes[:10]
    for scene in tqdm(scenes):

        gt_path = os.path.join(pcl_path, scene)
        loader = torch.load(gt_path)

        sem_gt, inst_gt = loader[2], loader[3]
        gtsem.append(np.array(sem_gt).astype(np.int32))
        gtinst.append(np.array(inst_gt).astype(np.int32))

        scene_path = os.path.join(data_path, scene)
        pred_mask = torch.load(scene_path)
        masks, category = pred_mask["ins"], pred_mask["class"]
        
        # score = torch.max(score, dim = -1)[0]

        n_mask = len(category)
        tmp = []
        for ind in range(n_mask):
            if isinstance(masks[ind], dict):
                mask = rle_decode(masks[ind])
            else:
                mask = (masks[ind] == 1).numpy().astype(np.uint8)
            conf = 1.0 # Normal OpenVocab
            # try:
            #     conf = score[ind].item() # CLIP-based OpenVocab
            # except:
            #     conf = score[ind] # CLIP-based OpenVocab

            final_class = float(category[ind])

            scene_id = scene.replace(".pth", "")
            tmp.append({"scan_id": scene_id, "label_id": final_class + 1, "conf": conf, "pred_mask": mask})

        res.append(tmp)

    scan_eval.evaluate(res, gtsem, gtinst)