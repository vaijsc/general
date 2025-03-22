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

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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


class Open3DIS_SAM_L2:
    def __init__(self, cfg):
        # Segment Anything
        self.device = "cuda:0"
        
        model_cfg = "sam2_hiera_l.yaml"
        self.predictor = build_sam2(model_cfg, cfg.foundation_model.sam2_checkpoint, device=self.device, apply_postprocessing=False)

        if cfg.data.dataset_name == 'scannet200':
            self.predictor = SAM2AutomaticMaskGenerator(
                model=self.predictor,    
                points_per_side=64,
                points_per_batch=128,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.92,
                stability_score_offset=0.7,
                crop_n_layers=1,
                box_nms_thresh=0.7,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=25.0,
                use_m2m=True,
            )
        elif cfg.data.dataset_name == 'scannetpp':
            self.predictor = SAM2AutomaticMaskGenerator(
                model = self.predictor,
                points_per_side = 64,
                points_per_batch = 128,
                pred_iou_thresh = 0.8,
                stability_score_thresh = 0.95,
                stability_score_offset = 1.0,
                mask_threshold = 0.0,
                box_nms_thresh = 0.7,
                crop_n_layers = 0,
                crop_n_points_downscale_factor = 1,
            ) # default configs

        self.cfg = cfg
        print('------- Loaded Segment Anything V2-------')

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
            
            self.visibility[mapping[:, 3] == 1] += 1

            dic = {"mapping": mapping.cpu(), "image": rgb_img, 'frame_id': frame_id}
            self.pcd_list.append(dic)

    def hierarchical_agglomerative_clustering(
        self,
        pcd_list,
        left,
        right,
        spp,
        n_spp,
        n_points,
        sieve,
        visi=0.7,
        reca = 1.0,
        simi=0.5,
        iterative=True,):
        '''
        Function from Open3DIS (2D-Guided-3D IPM)
        '''
        # global num_point, dc_feature_matrix, dc_feature_spp
        if left == right:
            device = spp.device
            # Graph initialization
            index = left

            if pcd_list[index]["masks"] is None:
                return [], []
            
            masks = pcd_list[index]["masks"].cuda()
            mapping = pcd_list[index]["mapping"].cuda()

            total_spp_points = torch_scatter.scatter((mapping[:, 3] == 1).float(), spp, dim=0, reduce="sum")

            weights = []
            ### Per mask processing
            mask3d = []

            for m, mask in enumerate(masks):
                spp_weights = torch.zeros((n_spp), dtype=torch.float32, device=device)
                idx = torch.nonzero(mapping[:, 3] == 1).view(-1)
                highlight_points = idx[
                    mask[mapping[idx][:, [1, 2]][:, 0], mapping[idx][:, [1, 2]][:, 1]].nonzero(as_tuple=True)[0]
                ].long()

                sieve_mask = torch.zeros((n_points), device=device)
                sieve_mask[highlight_points] = 1

                num_related_points = torch_scatter.scatter(sieve_mask.float(), spp, dim=0, reduce="sum")

                spp_weights = torch.where(
                    total_spp_points==0, 0, num_related_points / total_spp_points
                )
                target_spp = torch.nonzero(spp_weights >= 0.5).view(-1)

                if len(target_spp) <= 1:
                    continue

                elif len(target_spp) == 1:

                    target_weight = torch.zeros_like(spp_weights)
                    target_weight[target_spp] = spp_weights[target_spp]

                    group_tmp = torch.zeros((n_spp), dtype=torch.int8, device=device)
                    group_tmp[target_spp] = 1

                    mask3d.append(group_tmp)
                    weights.append(spp_weights)

                else:
                    pairwise_dc_dist = self.dc_feature_matrix[target_spp, :][:, target_spp]
                    pairwise_dc_dist[torch.eye((len(target_spp)), dtype=torch.bool, device=self.dc_feature_matrix.device)] = -10
                    max_dc_dist = torch.max(pairwise_dc_dist, dim=1)[0]

                    valid_spp = max_dc_dist >= 0.5

                    if valid_spp.sum() > 0:
                        target_spp = target_spp[valid_spp]

                        target_weight = torch.zeros_like(spp_weights)
                        target_weight[target_spp] = spp_weights[target_spp]

                        group_tmp = torch.zeros((n_spp), dtype=torch.int8, device=device)
                        group_tmp[target_spp] = 1

                        mask3d.append(group_tmp)
                        weights.append(spp_weights)

            if len(mask3d) == 0:
                return [], []
            mask3d = torch.stack(mask3d, dim=0)
            weights = torch.stack(weights, dim=0)
            return mask3d, weights

        mid = int((left + right) / 2)
        graph_1_onehot, weight_1 = self.hierarchical_agglomerative_clustering(
            pcd_list, left, mid, spp, n_spp, n_points, sieve, visi = visi, reca = reca, simi = simi, iterative=iterative
        )
        graph_2_onehot, weight_2 = self.hierarchical_agglomerative_clustering(
            pcd_list, mid + 1, right, spp, n_spp, n_points, sieve, visi = visi, reca = reca, simi = simi, iterative=iterative
        )

        if len(graph_1_onehot) == 0 and len(graph_2_onehot) == 0:
            return [], []

        if len(graph_1_onehot) == 0:
            return graph_2_onehot, weight_2

        if len(graph_2_onehot) == 0:
            return graph_1_onehot, weight_1

        if iterative:
            new_graph = torch.cat([graph_1_onehot, graph_2_onehot], dim=0)
            new_weight = torch.cat([weight_1, weight_2], dim=0)

            graph_feat = new_graph.bool().float() @ self.dc_feature_spp  # n, f

            graph_feat_matrix = pairwise_cosine_similarity(graph_feat, graph_feat)

            iou_matrix, _, recall_matrix = compute_relation_matrix_self(new_graph, spp, sieve)
            # iou_matrix, _, recall_matrix = compute_relation_matrix_self(new_graph)
            
            #####
            adjacency_matrix = (iou_matrix >= visi)
            if reca < 0.98:
                adjacency_matrix |= (recall_matrix >= reca)    
            if simi > 0.1: # scannetpp using 3D features from 3D backbone pretrained scannet200 yeilds not good results
                adjacency_matrix &= (graph_feat_matrix >= simi)
            adjacency_matrix = adjacency_matrix | adjacency_matrix.T
            #####

            # if adjacency_matrix
            if adjacency_matrix.sum() == new_graph.shape[0]:
                return new_graph, new_weight

            # merge instances based on the adjacency matrix
            connected_components = find_connected_components(adjacency_matrix)
            M = len(connected_components)

            merged_instance = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.int8, device=graph_2_onehot.device)
            merged_weight = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.float, device=graph_2_onehot.device)

            for i, cluster in enumerate(connected_components):
                merged_instance[i] = new_graph[cluster].sum(0)
                merged_weight[i] = new_weight[cluster].mean(0)

            new_graph = merged_instance
            new_weight = merged_weight

            return new_graph, new_weight
        
        new_graph, new_weight = [], [] 

        vis1 = torch.zeros((graph_1_onehot.shape[0]), device=graph_2_onehot.device)
        vis2 = torch.zeros((graph_2_onehot.shape[0]), device=graph_2_onehot.device)

        intersections = graph_1_onehot[:, spp].float() @ graph_2_onehot[:, spp].float().T
        # ious = intersections / ()
        # intersections = ((torch.logical_and(graph_1_onehot.bool()[:, None, :], graph_2_onehot.bool()[None, :, :])) * num_point[None, None]).sum(dim=-1)
        ious = intersections / ((graph_1_onehot.long() * self.num_point).sum(1)[:, None] + (graph_2_onehot.long() * num_point).sum(1)[None, :] - intersections)
        
        # similar_matrix = F.cosine_similarity(graph_1_feat[:, None, :], graph_2_feat[None, :, :], dim=2)
        graph_1_feat = torch.einsum('pn,nc->pc', graph_1_onehot.float(), self.dc_feature_spp) #/ torch.sum(graph_1_onehot, dim=1, keepdim=True)
        graph_2_feat = torch.einsum('pn,nc->pc', graph_2_onehot.float(), self.dc_feature_spp) #/ torch.sum(graph_2_onehot, dim=1, keepdim=True)
        similar_matrix = pairwise_cosine_similarity(graph_1_feat, graph_2_feat)
        
        row_inds = torch.arange((ious.shape[0]), dtype=torch.long, device=graph_1_onehot.device)
        max_ious, col_inds = torch.max(ious, dim=-1)
        valid_mask = (max_ious > visi) & (similar_matrix[row_inds, col_inds] > simi)
        
        row_inds_ = row_inds[valid_mask]
        col_inds_ = col_inds[valid_mask]
        vis2[col_inds_] = 1
        vis1[row_inds_] = 1

        union_masks = (graph_1_onehot[row_inds_] + graph_2_onehot[col_inds_]).int()
        intersection_masks = (graph_1_onehot[row_inds_] * graph_2_onehot[col_inds_]).bool()

        union_weight = 0.5 * (weight_1[row_inds_] + weight_2[col_inds_]) * intersection_masks \
                    + weight_1[row_inds_] * graph_1_onehot[row_inds_] \
                    + weight_2[col_inds_] * graph_2_onehot[col_inds_] 
        
        temp = (intersection_masks.float() + graph_1_onehot[row_inds_].float() + graph_2_onehot[col_inds_].float())
        union_weight = torch.where(temp == 0, 0, union_weight / temp)

        new_graph.append(union_masks.bool())
        new_weight.append(union_weight)

        nomatch_inds_group1 = torch.nonzero(vis1 == 0).view(-1)
        new_graph.append(graph_1_onehot[nomatch_inds_group1])
        new_weight.append(weight_1[nomatch_inds_group1])

        nomatch_inds_group2 = torch.nonzero(vis2 == 0).view(-1)
        new_graph.append(graph_2_onehot[nomatch_inds_group2])
        new_weight.append(weight_2[nomatch_inds_group2])


        new_graph = torch.cat(new_graph, dim=0)
        new_weight = torch.cat(new_weight, dim=0)

        return new_graph, new_weight        

    def generate3dproposal(self, scene_id, cfg):
        # Path
        exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)
        spp_path = os.path.join(cfg.data.spp_path, f"{scene_id}.pth")
        dc_feature_path = os.path.join(cfg.data.dc_features_path, scene_id + ".pth")

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

        dc_feature = loader.read_feature(dc_feature_path)
        dc_feature_spp = torch_scatter.scatter(dc_feature, spp, dim=0, reduce="sum")
        self.dc_feature_spp = F.normalize(dc_feature_spp, dim=1, p=2)
        self.dc_feature_matrix = pairwise_cosine_similarity(self.dc_feature_spp, self.dc_feature_spp)

        
        sieve_of_spp = [] # number of point in spp for fast calculating IoU between 3D masks
        for i in range (n_spp):
            sieve_of_spp.append((spp == i).sum().item()) 
        sieve_of_spp = torch.tensor(sieve_of_spp)

        # Compute Mapping
        self.visibility = torch.zeros((self.n_points), dtype=torch.int, device=spp.device)
        self.loader_2d_dict(pointcloud_mapper, loader)
        
        proposal_bank = []
        mean_spp = []
        for spp_id in (unique_spp):
            mean_spp.append(self.points[spp==spp_id].mean(dim=0))
        mean_spp = torch.stack(mean_spp)

        # 2D mask generator
        mask2d_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output, scene_id + '.pth')
        if os.path.exists(mask2d_path):
            grounded_data_dict = torch.load(mask2d_path)
        else:
            grounded_data_dict = {}
            for i in trange(len(self.pcd_list)):
                frame_id = self.pcd_list[i]['frame_id']
                image_rgb = self.pcd_list[i]['image']
                image_sam = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
                mapping = self.pcd_list[i]['mapping']
                masks = self.predictor.generate(image_sam)
                masks = [torch.tensor(id['segmentation']) for id in masks]
                masks = torch.stack(masks)
                if masks == None:  # No mask in the view
                    continue
                grounded_data_dict[frame_id] = {
                    "masks": masks_to_rle(masks),
                }
                if False:
                    # draw output image
                    image = image_rgb
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    for mask in masks:
                        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                    plt.axis("off")
                    # plot out
                    os.makedirs("../debug/" + scene_id, exist_ok=True)
                    plt.savefig(
                        os.path.join("../debug/" + scene_id + "/sam_" + str(i) + ".jpg"),
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=0.0,
                    )
            torch.save(grounded_data_dict, mask2d_path)
        
        # Generate 3D proposals from 2D masks using Open3DIS
        for i in trange(len(self.pcd_list)):
            frame_id = self.pcd_list[i]['frame_id']
            image_rgb = self.pcd_list[i]['image']
            image_sam = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
            mapping = self.pcd_list[i]['mapping']
            encoded_masks = grounded_data_dict[frame_id]['masks']
            masks = []
            for mask in encoded_masks:
                masks.append(torch.tensor(pycocotools.mask.decode(mask)))
            masks = torch.stack(masks, dim=0).cpu() # cuda fast but OOM
            self.pcd_list[i]['masks'] = masks

        # ScanNet200:
        if cfg.data.dataset_name == 'scannet200': 
            visi = 0.9 # iou
            recall = 0.9 # recall
            simi = 0.9 # dc_feats
            point_visi = 0.2
            valid_points = 50
        elif cfg.data.dataset_name == 'scannetpp':
            visi = 0.7 # iou
            recall = 1.0 # recall
            simi = 0.0 # dc_feats
            point_visi = 0.2
            valid_points = 50

        groups, weights = self.hierarchical_agglomerative_clustering(self.pcd_list, 0, len(self.pcd_list) - 1, spp, n_spp, self.n_points, sieve_of_spp, visi=visi, reca = recall, simi=simi, iterative=True)

        if len(groups) == 0:
            return None, None

        confidence = (groups.bool() * weights).sum(dim=1) / groups.sum(dim=1)
        groups = groups.to(torch.int64).cpu()

        spp = spp.cpu()
        proposals_pred = groups[:, spp]  # .bool()
        del groups, weights
        torch.cuda.empty_cache()

        ## These lines take a lot of memory # achieveing in paper result-> unlock this
        if point_visi > 0:
            start = 0
            end = proposals_pred.shape[0]
            inst_visibility = torch.zeros_like(proposals_pred, dtype=torch.float64).cpu()
            bs = 1000
            while(start<end):
                inst_visibility[start:start+bs] = (proposals_pred[start:start+bs] / self.visibility.clip(min=1e-6)[None, :].cpu().to(torch.float64))
                start += bs
            torch.cuda.empty_cache()    
            proposals_pred[inst_visibility < point_visi] = 0
        else: # pointvis==0.0
            pass
        
        proposals_pred = proposals_pred.bool()

        if point_visi > 0:
            proposals_pred_final = custom_scatter_mean(
                proposals_pred,
                spp[None, :].expand(len(proposals_pred), -1),
                dim=-1,
                pool=True,
                output_type=torch.float64,
            )
            proposals_pred = (proposals_pred_final >= 0.5)[:, spp]

        ## Valid points
        mask_valid = proposals_pred.sum(1) > 50
        proposals_pred = proposals_pred[mask_valid].cpu()
        
        return proposals_pred

