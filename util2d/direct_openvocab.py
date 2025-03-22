import os
import cv2
import numpy as np
import open3d as o3d
import pycocotools
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import matplotlib.pyplot as plt   
from PIL import Image
from tqdm import tqdm, trange

from util2d.openai_clip import CLIP_OpenAI
import clip

from loader3d import build_dataset
from loader3d.scannet_loader import scaling_mapping
from util3d.mapper import PointCloudToImageMapper
from util2d.util import show_mask_video, show_mask, masks_to_rle
from util3d.gen3d_utils import (
    compute_projected_pts,
    compute_projected_pts_torch,
    compute_relation_matrix_self,
    compute_relation_matrix_self_mem,
    compute_visibility_mask,
    compute_visibility_mask_torch,
    compute_visible_masked_pts,
    compute_visible_masked_pts_torch,
    find_connected_components,
    resolve_overlapping_masks,
    custom_scatter_mean,
)
from torchmetrics.functional import pairwise_cosine_similarity
from util3d.pointnet2.pointnet2_utils import furthest_point_sample
import pickle

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

class OpenVocab_Direct:
    def __init__(self, cfg, class_names):
        # OpenAI CLIP
        self.device = "cuda:0"
        self.clip_model = CLIP_OpenAI(cfg)
        self.clip_dim = self.clip_model.dim
        self.cfg = cfg
        self.class_names = class_names
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.clip_adapter.encode_text(clip.tokenize(class_names).to(self.device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        print('------- Loaded OpenAI CLIP-------')

    def loader_2d_dict(self, pointcloud_mapper, loader, interval=2):
        '''''
        Align 3D point cloud with 2D images
        '''''
        self.pcd_list = []
        print("-------Compute Mapping 2D-3D-------")
        
        img_dim = self.cfg.data.img_dim
        for i in trange(0, len(loader), interval):
            frame = loader[i]
            frame_id = frame["frame_id"]

            pose = loader.read_pose(frame["pose_path"])
            depth = loader.read_depth(frame["depth_path"])
            rgb_img = loader.read_image(frame["image_path"])
            rgb_img_dim = rgb_img.shape[:2]
                
            if "scannetpp" in self.cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
                depth = cv2.resize(depth, (img_dim[0], img_dim[1]))
                mapping = torch.ones([self.n_points, 4], dtype=int, device=self.points.device)
                mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, self.points, depth, intrinsic = frame["translated_intrinsics"])
            elif "scannet200" in self.cfg.data.dataset_name:
                mapping = torch.ones([self.n_points, 4], dtype=int, device=self.points.device)
                mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, self.points, depth, intrinsic = frame["scannet_depth_intrinsic"])
                new_mapping = scaling_mapping(
                    torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[0], rgb_img_dim[1]
                )
                mapping[:, 1:4] = torch.cat((new_mapping, mapping[:, 3].unsqueeze(1)), dim=1)
            else:
                raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")
            
            dic = {"mapping": mapping.cpu(), "image": rgb_img, 'frame_id': frame_id}
            self.pcd_list.append(dic)      

    def get_final_instance(self, scene_id, cfg):

        # Path
        exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)
        spp_path = os.path.join(cfg.data.spp_path, f"{scene_id}.pth")

        # Data Loader
        scene_dir = os.path.join(cfg.data.datapath, scene_id)
        loader = build_dataset(root_path=scene_dir, cfg=cfg)

        pointcloud_mapper = PointCloudToImageMapper(
            image_dim=self.cfg.data.img_dim, intrinsics=loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
        )

        # Point, Superpoint, 3D Features
        self.points = loader.read_pointcloud()
        self.points = torch.from_numpy(self.points).to(self.device)
        self.n_points = self.points.shape[0]
        
        spp = loader.read_spp(spp_path)
        unique_spp, spp, self.num_point = torch.unique(spp, return_inverse=True, return_counts=True)
        n_spp = len(unique_spp)

        self.spp = spp
        self.n_spp = n_spp
        self.loader_2d_dict(pointcloud_mapper, loader)
        
        save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output, scene_id + '.pth')        
        dic = torch.load(save_dir_cluster)
        instance3d = dic['ins']
        try:
            instance3d = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance3d])
        except:
            pass  
        mask_bank, id_bank = dic['mask2d_bank'], dic['id_bank']
        feature_bank = []

        for rr in trange(instance3d.shape[0]):
            msk_set = mask_bank[rr]
            images = [self.pcd_list[id]['image'] for id in id_bank[rr]]
            mappings = [self.pcd_list[id]['mapping'] for id in id_bank[rr]]

           #NOTE: Multiscale Crops
            kexp = 0.2
            H, W = images[0].shape[0], images[0].shape[1]
            cropped_regions = []
            feature_weight = []
            # Open Vocab
            for i in range(len(images)): # 5candidate
                img = np.array(images[i])
                # intersect = torch.logical_and(mappings[i][:,3]==1, instance3d[rr]).sum()xxx
                msk = torch.tensor(msk_set[i])
                if False:
                    # draw output image
                    image = img
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    show_mask(msk.cpu().numpy(), plt.gca(), random_color=True)
                    plt.axis("off")
                    # plot out
                    os.makedirs("../debug/scannetpp/" + scene_id, exist_ok=True)
                    plt.savefig(
                        os.path.join("../debug/scannetpp/" + scene_id + "/sam_" + str(i) + ".jpg"),
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=0.0,
                    )
                rows, cols = torch.where(msk[0] == True)
                if rows.shape[0] == 0 or cols.shape[0] == 0:
                    continue
                x1, y1 = rows.min().item(), cols.min().item()
                x2, y2 = rows.max().item(), cols.max().item()
                
                if x2 - x1 == 0 or y2 - y1 == 0:
                    continue
                

                for round in range(3):
                    cropped_image = img[x1:x2, y1:y2,:]
                    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                        continue
                    im = Image.fromarray(cropped_image)
                    row, col = torch.where(msk[0,x1:x2, y1:y2] == False)
                    tmp = torch.tensor(np.asarray(im)).to(self.device)
                    # Blurring background - trick here improve CLIP feature
                    tmp[row, col, 0] = (0 * 0.5 + tmp[row, col, 0] * (1 - 0.5)).to(torch.uint8)
                    tmp[row, col, 1] = (0 * 0.5 + tmp[row, col, 1] * (1 - 0.5)).to(torch.uint8)
                    tmp[row, col, 2] = (0 * 0.5 + tmp[row, col, 2] * (1 - 0.5)).to(torch.uint8)                    
                    im1 = Image.fromarray((tmp.cpu().numpy()))
                    cropped_regions.append(self.clip_model.clip_preprocess(im1))

                    # if i < 10 and rr == 3: 
                    #     os.makedirs(f"debug/debug{rr}", exist_ok=True)
                    #     im1.save(f"debug/debug{rr}/image{i}_crop{round}_size_{im1.size}.jpg")
                    # feature_weight.append(intersect)
                    tmpx1 = int(max(0, x1 - (x2 - x1) * kexp * round))
                    tmpy1 = int(max(0, y1 - (y2 - y1) * kexp * round))
                    tmpx2 = int(min(H - 1, x2 + (x2 - x1) * kexp * round))
                    tmpy2 = int(min(W - 1, y2 + (y2 - y1) * kexp * round))
                    x1, y1, x2, y2 = tmpx1, tmpy1, tmpx2, tmpy2

            mask_feature = torch.zeros((1, self.clip_dim), dtype = torch.float16).to(self.device)
            if len(cropped_regions) != 0:
                crops = torch.stack(cropped_regions).cuda()
                img_batches = torch.split(crops, 64, dim = 0)
                image_features = []
                with torch.no_grad(), torch.cuda.amp.autocast():
                    for img_batch in img_batches:
                        image_feat = self.clip_model.clip_adapter.encode_image(img_batch)
                        image_feat /= image_feat.norm(dim = -1, keepdim = True)
                        image_features.append(image_feat)
                image_features = torch.cat(image_features, dim = 0) # (n, 768)
                # feature_weight = torch.tensor(feature_weight)
                # mask_feature = (feature_weight.unsqueeze(0).to(self.device).to(torch.float16) @ image_features)/torch.sum(feature_weight, -1)
                mask_feature = torch.sum(image_features.to(torch.float16).to(self.device), dim=0).unsqueeze(0)
                breakpoint() #NOTE here Consider NORMING
                # mask_feature = torch.max(image_features, dim=0, keepdim=True)[0].to(torch.float16).to(self.device)
            feature_bank.append(mask_feature)            

        class_ids = []
        categories = []
        feature_bank = torch.stack(feature_bank).squeeze(1).to(self.device) # nmask, 768
        predicted_class = (300.0 * feature_bank @ self.text_features.cuda().T.float()).softmax(dim=-1)
        idx = torch.argmax(predicted_class, dim = 1)  
        cls_final = idx.cpu()
        categories = [self.class_names[tt.item()] for tt in cls_final]

        return instance3d, cls_final, categories

