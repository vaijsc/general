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

class OpenVocab_SPPWise:
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

    def superpoint_aggregator(self, scene_id):
        '''
            CLIP feature aggregator using weighted Superpoints
        '''
        pcd_list = self.pcd_list
        print("-------Feature Aggregating-------")
        feature_bank = torch.zeros((self.n_spp, self.clip_dim), dtype = torch.float64, device = self.device)
        mappings = []
        images = []
        for dic in self.pcd_list:
            mappings.append(dic['mapping'])
            images.append(dic['image'])
        mappings = torch.stack(mappings)
        cropped_regions = []
        cropped_3d = []

        instance3d_spp = custom_scatter_mean(
            self.instance3d.to(self.device),
            self.spp[None, :].expand(len(self.instance3d), -1),
            dim=-1,
            pool=True,
            output_type=torch.float64,
            )
        instance3d_spp = (instance3d_spp >= 0.5)

        for (obj_id, inst) in enumerate(tqdm(self.instance3d)):
            
            target = self.storage2d[obj_id]
            msk_set = target[0]['mask2d']
            img_ids = target[0]['im_mask2d_id']
            images = [self.pcd_list[id]['image'] for id in img_ids]
            mappings = [self.pcd_list[id]['mapping'] for id in img_ids]

            # Obtaining top-k views
            n_views = len(mappings)
            mappings_all = torch.stack(mappings)
            conds = (mappings_all[..., 3] == 1) & inst[None].expand(n_views, -1)  # n_view, n_points
            count_views = conds.sum(1)
            valid_count_views = count_views > 20
            valid_inds = torch.nonzero(valid_count_views).view(-1)
            if len(valid_inds) == 0:
                continue
            topk_counts, topk_views = torch.topk(
                count_views[valid_inds], k=min(10, len(valid_inds)), largest=True
            )
            topk_views = valid_inds[topk_views]    

            images = [images[indice] for indice in topk_views]
            mappings = [mappings[indice] for indice in topk_views]
            msk_set = [msk_set[indice] for indice in topk_views]

            # Multiscale image crop from topk views
            for v in range(len(images)):
                img = np.array(images[v])
                msk = torch.tensor(msk_set[v])            

                # Calculate the bounding rectangle
                H, W = images[0].shape[0], images[0].shape[1]
                rows, cols = torch.where(msk[0] == True)
                if rows.shape[0] == 0 or cols.shape[0] == 0:
                    continue
                x1, y1 = rows.min().item(), cols.min().item()
                x2, y2 = rows.max().item(), cols.max().item()
                kexp = 0.2
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
                    cropped_3d.append(instance3d_spp[obj_id])
                    # if i < 10 and rr == 3: 
                    #     os.makedirs(f"debug/debug{rr}", exist_ok=True)
                    #     im1.save(f"debug/debug{rr}/image{i}_crop{round}_size_{im1.size}.jpg")
                    # feature_weight.append(intersect)
                    tmpx1 = int(max(0, x1 - (x2 - x1) * kexp * round))
                    tmpy1 = int(max(0, y1 - (y2 - y1) * kexp * round))
                    tmpx2 = int(min(H - 1, x2 + (x2 - x1) * kexp * round))
                    tmpy2 = int(min(W - 1, y2 + (y2 - y1) * kexp * round))
                    x1, y1, x2, y2 = tmpx1, tmpy1, tmpx2, tmpy2

            # Batch forwarding CLIP features
        if len(cropped_regions) != 0:
            crops = torch.stack(cropped_regions).cuda()
            img_batches = torch.split(crops, 64, dim=0)
            image_features = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for img_batch in img_batches:
                    image_feat = self.clip_model.clip_adapter.encode_image(img_batch)
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                    image_features.append(image_feat)
            image_features = torch.cat(image_features, dim=0)
            cropped_3d = torch.stack(cropped_3d).to(torch.float16)
            feature_bank += torch.einsum('ij,ik->jk',cropped_3d.to(self.device), image_features) 
        

        ################################## 3D only ##################################
        mappings = []
        images = []
        for dic in self.pcd_list:
            mappings.append(dic['mapping'])
            images.append(dic['image'])
        data_path3D = "/root/3dllm/minhlnh/FreeVocab-3DIS/data/Scannet200/Scannet200_3D/class_ag_res_200_isbnetfull"
        scene_path3D = os.path.join(data_path3D, scene_id+'.pth')
        pred_mask3D = torch.load(scene_path3D)
        masks3D = pred_mask3D['ins']
        masks3D = torch.tensor(masks3D)
        # masks3D = torch.cat([self.instance3d, torch.tensor(masks3D)], 0)   
        masks3D_spp = custom_scatter_mean(
            masks3D.to(self.device),
            self.spp[None, :].expand(len(masks3D), -1),
            dim=-1,
            pool=True,
            output_type=torch.float64,
            )
        masks3D_spp = (masks3D_spp >= 0.5)        
        mappings = torch.stack(mappings, dim=0)
        n_views = len(mappings)
        cropped_regions = []
        cropped_3d = []
        
        for inst in trange(masks3D.shape[0]):
            # Obtaining top-k views
            conds = (mappings[..., 3] == 1) & masks3D[inst][None].expand(n_views, -1)  # n_view, n_points
            count_views = conds.sum(1)
            valid_count_views = count_views > 20
            valid_inds = torch.nonzero(valid_count_views).view(-1)
            if len(valid_inds) == 0:
                continue
            topk_counts, topk_views = torch.topk(
                count_views[valid_inds], k=min(1, len(valid_inds)), largest=True
            )
            topk_views = valid_inds[topk_views]

            # Multiscale image crop from topk views
            for v in topk_views:
                point_inds_ = torch.nonzero((mappings[v][:, 3] == 1) & (masks3D[inst] == 1)).view(-1)
                projected_points = torch.tensor(mappings[v][point_inds_][:, [1, 2]]).cuda()
                # Calculate the bounding rectangle
                mi = torch.min(projected_points, axis=0)
                ma = torch.max(projected_points, axis=0)
                x1, y1 = mi[0][0].item(), mi[0][1].item()
                x2, y2 = ma[0][0].item(), ma[0][1].item()

                if x2 - x1 == 0 or y2 - y1 == 0:
                    continue
                # Multiscale clip crop follows OpenMask3D
                kexp = 0.2
                H, W = images[v].shape[0], images[v].shape[1]
                ## 3 level cropping
                for round in range(3):
                    cropped_image = images[v][x1:x2, y1:y2, :]
                    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                        continue
                    cropped_regions.append(self.clip_model.clip_preprocess(Image.fromarray(cropped_image)))
                    cropped_3d.append(masks3D_spp[inst])

                    # reproduce from OpenMask3D
                    tmpx1 = int(max(0, x1 - (x2 - x1) * kexp * round))
                    tmpy1 = int(max(0, y1 - (y2 - y1) * kexp * round))
                    tmpx2 = int(min(H - 1, x2 + (x2 - x1) * kexp * round))
                    tmpy2 = int(min(W - 1, y2 + (y2 - y1) * kexp * round))
                    x1, y1, x2, y2 = tmpx1, tmpy1, tmpx2, tmpy2

        # Batch forwarding CLIP features
        if len(cropped_regions) != 0:
            crops = torch.stack(cropped_regions).cuda()
            img_batches = torch.split(crops, 64, dim=0)
            image_features = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for img_batch in img_batches:
                    image_feat = self.clip_model.clip_adapter.encode_image(img_batch)
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                    image_features.append(image_feat)
            image_features = torch.cat(image_features, dim=0)
            cropped_3d = torch.stack(cropped_3d).to(torch.float16)
            feature_bank += torch.einsum('ij,ik->jk',cropped_3d.to(self.device), image_features) 

        feature_bank = F.normalize(feature_bank, dim=1, p=2)                  
        #################
        # for track in tqdm(self.storage2d):
        #     image_id = track['image_id']
        #     video_mask = track['video_mask']
        #     images = []
        #     lifted3d = []
        #     masks = []
        #     track_frame = 0
        #     spp_weights = torch.zeros((self.n_spp), dtype=torch.float32, device=self.device)
        #     sieve_mask = torch.zeros((self.n_points), device=self.device)
            
        #     for id in image_id:
        #         images.append(pcd_list[id]['image'])
        #         masks.append(video_mask[track_frame][1])
        #         mapping = pcd_list[id]['mapping'].to(self.device)
        #         mask2d = torch.tensor(video_mask[track_frame][1])[0].to(self.device)
        #         track_frame += 1
        #         total_spp_points = torch_scatter.scatter((mapping[:, 3] == 1).float(), self.spp.to(self.device), dim=0, reduce="sum")
        #         idx = torch.nonzero(mapping[:, 3] == 1).view(-1)
        #         highlight_points = idx[
        #             mask2d[mapping[idx][:, [1, 2]][:, 0], mapping[idx][:, [1, 2]][:, 1]].nonzero(as_tuple=True)[0]
        #         ].long()

        #         spp_weights[:] = 0.0
        #         sieve_mask[:] = 0.0
        #         sieve_mask[highlight_points] = 1

        #         num_related_points = torch_scatter.scatter(sieve_mask.float(), self.spp.to(self.device), dim=0, reduce="sum")

        #         spp_weights = torch.where(
        #             total_spp_points==0, 0, num_related_points / total_spp_points
        #         )
        #         lifted3d.append(spp_weights)

        #     #NOTE: Multiscale Crops
        #     kexp = 0.2
        #     H, W = images[0].shape[0], images[0].shape[1]
        #     cropped_regions = []
        #     cropped_3d = []
        #     # Open Vocab
        #     for i in range(len(images)):
        #         img = np.array(images[i])
        #         msk = torch.tensor(masks[i])
        #         rows, cols = torch.where(msk[0] == True)
        #         if rows.shape[0] == 0 or cols.shape[0] == 0:
        #             continue
        #         x1, y1 = rows.min().item(), cols.min().item()
        #         x2, y2 = rows.max().item(), cols.max().item()

        #         for round in range(5):
        #             cropped_image = img[x1:x2, y1:y2,:]
        #             if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        #                 continue
        #             cropped_regions.append(self.clip_model.clip_preprocess(Image.fromarray(cropped_image)))
        #             cropped_3d.append(lifted3d[i])
        #             tmpx1 = int(max(0, x1 - (x2 - x1) * kexp * round))
        #             tmpy1 = int(max(0, y1 - (y2 - y1) * kexp * round))
        #             tmpx2 = int(min(H - 1, x2 + (x2 - x1) * kexp * round))
        #             tmpy2 = int(min(W - 1, y2 + (y2 - y1) * kexp * round))
        #             x1, y1, x2, y2 = tmpx1, tmpy1, tmpx2, tmpy2
            
        #     if len(cropped_regions) != 0:
        #         crops = torch.stack(cropped_regions).cuda()
        #         img_batches = torch.split(crops, 64, dim = 0)
        #         image_features = []
        #         with torch.no_grad(), torch.cuda.amp.autocast():
        #             for img_batch in img_batches:
        #                 image_feat = self.clip_model.clip_adapter.encode_image(img_batch)
        #                 image_feat /= image_feat.norm(dim = -1, keepdim = True)
        #                 image_features.append(image_feat)
        #         image_features = torch.cat(image_features, dim = 0) # (n, 768)
        #         cropped_3d = torch.stack(cropped_3d)
        #         feature_bank += torch.einsum('ij,ik->jk',cropped_3d, image_features) 
        # feature_bank = F.normalize(feature_bank, dim=1, p=2)           
        return feature_bank

    def refine_openvocab(self, scene_id, cfg, genfeature = True):
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
        
        #NOTE: 2D preparation
        save_dir_2d = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output, scene_id)
        file_names = os.listdir(save_dir_2d)
        file_names.sort()
        
        print("-------Prepare 2D Files-------")
        self.storage2d = []
        for filename in tqdm(file_names):
            file = open(os.path.join(save_dir_2d, filename), "rb")
            self.storage2d.append(pickle.load(file))

        save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output, scene_id + '.pth')        
        dic = torch.load(save_dir_cluster)
        self.instance3d = dic['ins']
        self.obj_id = dic['obj_bank']
        try:
            self.instance3d = torch.stack([torch.tensor(rle_decode(ins)) for ins in self.instance3d])
        except:
            pass        

        if genfeature:
            self.feature_bank = self.superpoint_aggregator(scene_id)
        else:
            self.feature_bank = torch.load(os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clip_feature, scene_id + '.pth'))['feat']
        
        return self.feature_bank

    def get_final_instance(self, scene_id, cfg):
        class_ids = []
        categories = []

        #NOTE: ADD 3D proposals
        data_path3D = "/root/3dllm/minhlnh/FreeVocab-3DIS/data/Scannet200/Scannet200_3D/class_ag_res_200_isbnetfull"
        scene_path3D = os.path.join(data_path3D, scene_id+'.pth')
        pred_mask3D = torch.load(scene_path3D)
        masks3D = pred_mask3D['ins']
        self.instance3d = torch.cat([self.instance3d, torch.tensor(masks3D)], 0)        

        instance3d_spp = custom_scatter_mean(
            self.instance3d.to(self.device),
            self.spp[None, :].expand(len(self.instance3d), -1),
            dim=-1,
            pool=True,
            output_type=torch.float64,
            )
        instance3d_spp = (instance3d_spp >= 0.5)

        predicted_class = torch.zeros((self.feature_bank.shape[0], self.text_features.shape[0]), dtype = torch.float32)
        bs = 100000
        for batch in range(0, self.feature_bank.shape[0], bs):
            start = batch
            end = min(start + bs, self.feature_bank.shape[0])
            predicted_class[start:end] = (300 * self.feature_bank[start:end].to(torch.float32).cpu() @ self.text_features.T.to(torch.float32).cpu()).softmax(dim=-1).cpu()
        predicted_class = predicted_class.cuda()

        del self.feature_bank
        torch.cuda.empty_cache() 
        # NOTE Mask-wise semantic scores
        inst_class_scores = torch.einsum("kn,nc->kc", instance3d_spp.float().cpu(), predicted_class.float().cpu()).cuda()  # K x classes
        inst_class_scores = inst_class_scores / instance3d_spp.float().cuda().sum(dim=1)[:, None]  # K x classes

        # # NOTE Top-K instances
        inst_class_scores = inst_class_scores.reshape(-1)  # n_cls * n_queries
        
        if "scannetpp" in self.cfg.data.dataset_name:
            num_classes = 84
        else:
            num_classes = 198

        labels = (
            torch.arange(num_classes, device=inst_class_scores.device)
            .unsqueeze(0)
            .repeat(instance3d_spp.shape[0], 1)
            .flatten(0, 1)
        )
        cur_topk = 600
        _, idx = torch.topk(inst_class_scores, k=min(cur_topk, len(inst_class_scores)), largest=True)
        mask_idx = torch.div(idx, num_classes, rounding_mode="floor")

        cls_final = labels[idx].cpu()
        scores_final = inst_class_scores[idx].cpu()
        masks_final = self.instance3d[mask_idx.cpu()]        

        categories = [self.class_names[tt.item()] for tt in cls_final]

        return masks_final, cls_final, categories

