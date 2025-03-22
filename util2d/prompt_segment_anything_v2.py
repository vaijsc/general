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

# SAM-2 Video-Image Predictor
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# from segment_anything import SamPredictor, build_sam, build_sam_hq

from scipy.sparse import csr_matrix
import pulp

from loader3d import build_dataset
from loader3d.scannet_loader import scaling_mapping
from util3d.mapper import PointCloudToImageMapper
from util2d.util import show_mask_video
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
    cost_LP,
    cost_LP_opt
)
from torchmetrics.functional import pairwise_cosine_similarity
from util3d.pointnet2.pointnet2_utils import furthest_point_sample, ball_query
from random import randrange
from scipy.spatial import KDTree

np.random.seed(0)
torch.manual_seed(0)

def euclidean_dist(point, centroids):
    dist = torch.sqrt(torch.sum((point - centroids) ** 2, dim=1))
    return dist

def furthest_sampling2d(data, num_centorids = 5):
    # cudarize for as fast as possible
    data = torch.tensor(data).cuda()
    idx = randrange(len(data))
    centroids = [data[idx]]
    for i in range(0, num_centorids):
        if i == idx:
            continue
        distances = []
        for x in data:
            # 1. Find distances between a point and all centroids
            dists = euclidean_dist(x, torch.stack(centroids, dim = 0))
            # 2. Save the min distance 
            distances.append(torch.min(dists))
        # this will be the new farthest centroid
        max_idx = torch.argmax(torch.tensor(distances))
        centroids.append(data[max_idx])
    return centroids



class Prompt_SAM_L2:
    def __init__(self, cfg):
        ### Segment Anything
        self.device = "cuda:0"
        
        model_cfg = "sam2_hiera_l.yaml"
        ### Video Predictor
        self.predictor = build_sam2_video_predictor(model_cfg, cfg.foundation_model.sam2_checkpoint, device=self.device)
        ### Image Predictor
        self.image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, cfg.foundation_model.sam2_checkpoint, device=self.device))
        
        ### SAM-HQ
        # sam_checkpoint = "../ckpts/sam_hq_vit_h.pth"
        # sam_hq = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(self.device)
        # self.image_predictor = SamPredictor(sam_hq)
        

        self.cfg = cfg
        print('------- Loaded Segment Anything V2 Image+Video-------')

    def spherical_query(self, anchor, mean_spp, neighbors=32):
        '''
            Spherical Query points around the anchor
            Binary Search radius for accumulating enough neighbors ~ KNN
        '''
        left = 0
        right = 1000
        for _ in range(64): # 2^{-64} decimal precision
            radius = (left + right)/2
            dist_2 = torch.sum((mean_spp - anchor)**2, dim=-1) # euclidean dist^2
            num = (dist_2 < radius**2).sum().item()
            if num < neighbors:
                left = radius
            else:
                right = radius
        
        return (dist_2 < radius**2)

    def kdtree_query(self, anchor, mean_spp, neighbors = 32):
        '''
            Scipy KDTree queries k neareas points around the anchor ~ spherical_query
            # NOTE: a bit slow, but yeilds worser result 
            (25.8->24.6) on the first scene (notyet stress test)
        '''
        kdtree = KDTree(mean_spp.detach().cpu().numpy())
        dists_ref, inds_ref = kdtree.query(anchor.unsqueeze(0).detach().cpu().numpy(), k = neighbors)
        target = torch.zeros(mean_spp.shape[0], dtype = torch.bool)
        target[inds_ref] = True
        return target

    def construct_video(self, loader, target):
        '''''
            Construct a video from start_id -> end_id, save it to video path
            Ouputing corresponding 2D-3D mapping list
        '''''
        images = []
        mappings = []
        for (i,id) in enumerate(target):
            image = loader[id]['image']
            # Image.fromarray(image).save(os.path.join(path,'{0:04}'.format(i)+'.jpg'))
            images.append(Image.fromarray(image))
            mappings.append(loader[id]['mapping'])
        return images, mappings

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
            rgb_img_dim = rgb_img.shape[:5]
                
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

            dic = {"mapping": mapping.cpu(), "image": rgb_img}
            self.pcd_list.append(dic)

    def generate3dproposal(self, scene_id, cfg, promptclick):
        # Path
        exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)
        spp_path = os.path.join(cfg.data.spp_path, f"{scene_id}.pth")
        dc_feature_path = os.path.join(cfg.data.dc_features_path, scene_id + ".pth")
        gt_pth = os.path.join(cfg.data.gt_pth, scene_id + ".pth")

        # Load 3D proposals
        gts = torch.load(gt_pth)
        total_gts = np.unique(gts[3])
        total_gts = total_gts[total_gts != -100]
        gt_proposals = []
        
        for tt in (total_gts[:]):
            gt_proposals.append(torch.tensor(gts[3]==tt))
        gt_proposals = gt_proposals[:]
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
        unique_spp, spp, num_point = torch.unique(spp, return_inverse=True, return_counts=True)
        self.spp = spp # global
        n_spp = len(unique_spp)

        dc_feature = loader.read_feature(dc_feature_path)
        dc_feature_spp = torch_scatter.scatter(dc_feature, spp, dim=0, reduce="sum")
        dc_feature_spp = F.normalize(dc_feature_spp, dim=1, p=2)
        dc_feature_matrix = pairwise_cosine_similarity(dc_feature_spp, dc_feature_spp)

        
        sieve_of_spp = [] # number of point in spp for fast calculating IoU between 3D masks
        for i in range (n_spp):
            sieve_of_spp.append((spp == i).sum().item()) 
        sieve_of_spp = torch.tensor(sieve_of_spp).to(self.device)

        # Compute Mapping
        self.loader_2d_dict(pointcloud_mapper, loader, interval = cfg.data.img_interval)
        
        proposal_bank = []
        mean_spp = []
        for spp_id in (unique_spp):
            mean_spp.append(self.points[spp==spp_id].mean(dim=0))
        mean_spp = torch.stack(mean_spp)
        mean_spp_origin = mean_spp.clone()
        
        # Fast Calculating the pointdistribution of each spp given an image
        # NOTE: shape = (nFrame, n_spp): point distribution of spp on every frame
        point_distribution = torch.zeros((len(self.pcd_list), n_spp), dtype = torch.int32, device = self.device)
        for id in range(len(self.pcd_list)):
            mapping = self.pcd_list[id]['mapping'].to(self.device)
            point_distribution[id] = torch_scatter.scatter_sum(mapping[:,3], spp)
        
        # Grow an object using point prompt spp
        N_Centorids = promptclick
        for gt_proposal in tqdm([gt_proposals[13]]):
            mask_flag = False
            ################################################################
            # Groundtruth proposal            
            target_spp = spp[gt_proposal].unique()
            target_id = []
            for t_spp in target_spp:
                #NOTE: robust spp
                # if (spp[gt_proposal] == t_spp).sum().item() == sieve_of_spp[t_spp].item():
                #     target_id.append(t_spp)
                #NOTE: just take it
                target_id.append(t_spp)
            target_id = torch.tensor(target_id).to(self.device)
            if target_id.shape[0] == 0:
                print('Empty proposal')
                proposal_bank.append(torch.zeros((self.n_points), dtype=torch.int8, device=self.device))
                continue
            # Sample number of spp accordingly to to # promptclick
            mean_spps = mean_spp[target_id]
            fps_inds = furthest_point_sample(mean_spps.to(torch.float).unsqueeze(0), N_Centorids).long()[0]
            spp_ids = np.array(fps_inds.cpu())
            spp_ids = target_id[spp_ids]            

            target_mask = torch.zeros((spp.shape[0])).to(self.device)
            for spp_id in (spp_ids):
                target_mask = torch.logical_or((spp == spp_id), target_mask)
            
            mask = target_mask
            # Currently, we calculate the spherical distribution of the first spp
            spp_id = spp_ids[0]
            # Calculate pixel' distribution on every frame given a spp  
            # NOTE: shape =  (nFrame)  
            pixel_distribution = point_distribution[:,spp_id]
            k_closest_spp = self.spherical_query(mean_spp_origin[spp_id], mean_spp_origin, neighbors = cfg.proposal3d.neighbors)

            ### K Neareast spherical Neighbors KDTree                
            # k_closest_spp = self.kdtree_query(mean_spp_origin[spp_id], mean_spp_origin, neighbors=32)

            ####### NOTE: weighted distribution #######
            weighted_distribution = torch.mean((point_distribution[:,k_closest_spp]/sieve_of_spp[k_closest_spp]), dim = -1)
            weighted_distribution *= pixel_distribution
            
            vals, peaks = torch.topk(weighted_distribution, k = 1)
            
            # Find superpoint interval
            for peak_id in peaks:
                peak_id = peak_id.item()
                peak_value = pixel_distribution[peak_id].item()
                if peak_value == 0:
                    continue

                ####### NOTE: track the whole video, time consuming
                left = 0
                right = pixel_distribution.shape[0] - 1


                # Backward Video 
                numback = len(torch.where(weighted_distribution[left:peak_id + 1] > 0.0)[0])
                target_backward = (peak_id - torch.topk(weighted_distribution[left:peak_id + 1], k = min(self.cfg.proposal3d.weighted_views, numback), largest = True)[1]).tolist()
                target_backward.sort()
                
                images1, mapping1 = self.construct_video(self.pcd_list[left:peak_id + 1][::-1], target = target_backward)
                
                # Forward Video
                numforward = len(torch.where(weighted_distribution[peak_id:right + 1] > 0.0)[0])
                target_forward = (torch.topk(weighted_distribution[peak_id:right + 1], k = min(self.cfg.proposal3d.weighted_views, numforward), largest = True)[1]).tolist()
                target_forward.sort()
                images2, mapping2 = self.construct_video(self.pcd_list[peak_id: right + 1], target = target_forward)

                images1 = images1 + images2
                mapping1 = mapping1 + mapping2

                # 2D sampling
                mapping = self.pcd_list[peak_id]['mapping'].to(self.device)                
                start_image = images1[0]
                start_image = np.array(start_image.convert("RGB"))
                self.image_predictor.set_image(start_image)

                RANSAC_Rounds = N_Centorids
                init_mask_prompts = None
                selected_points = []
                while RANSAC_Rounds > 0:
                    # NOTE: PFS sample
                    RANSAC_Rounds -= 1
                    root_spp = spp_ids[RANSAC_Rounds]
                    root_mask = (spp == root_spp)
                    intersect = torch.logical_and(root_mask, mapping[:,3])
                    if intersect.sum().item() == 0:
                        continue
                    selected_points.append(torch.stack(furthest_sampling2d(mapping[intersect][:,[2,1]], num_centorids = 2)))

                if len(selected_points) == 0:
                    continue
                points_prompt =  np.array(torch.cat(selected_points).cpu())
                
                # SAM-2 image predictor
                labels = np.array([1]*points_prompt.shape[0], np.int32)
                masks_prompts, scores, logits = self.image_predictor.predict(
                    point_coords=points_prompt,
                    point_labels=labels,
                    multimask_output=False,
                )
                init_mask_prompts = masks_prompts
                maximum_score = scores[0]
                

                # # The anchored mask has to be good enough
                # if maximum_score < 0.85:
                #     continue

                # Inline SAM2 Forward function 
                def forward_sam2(images, mappings_video, video_path = None):
                    '''''
                    Returning set of 3D proposals corresponding to mask of the video by SAM-2
                    using SPP lifting as Open3DIS
                    '''''
                    mapping = self.pcd_list[peak_id]['mapping'].to(self.device)
                    intersect = torch.logical_and(mask, mapping[:,3])
                    inference_state = self.predictor.init_state(images_list=images) # samtrack3d loader
                    self.predictor.reset_state(inference_state)
                    ann_frame_idx = 0  # the frame index we interact with
                    ann_obj_id = 1  # give a unique id to each object we interact with
                    
                    labels = np.array([1], np.int32) # for labels, `1` means positive click and `0` means negative click

                    ###################### Mask Prompt ######################
                    ### Get masks having highest score ###
                    masks_prompts = init_mask_prompts

                    for mask_id, mk_prompt in enumerate(masks_prompts):
                        ann_frame_idx = 0  # the frame index we interact with
                        ann_obj_id = mask_id + 1  # give a unique id to each object we interact with

                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            mask=mk_prompt,
                        )
                    ###################### Point 3D aware refining Prompt ######################
                    for frame_id in range(1, len(mappings_video), max(1, int(len(mappings_video)/5))):
                        # NOTE same prompt # selected_points = furthest_sampling2d(mapping[intersect][:,[2,1]].cpu(), num_centorids = 3)
                        mapping = mappings_video[frame_id].to(self.device)
                        intersect = torch.logical_and(mask, mapping[:,3])
                        selected_points = furthest_sampling2d(mapping[intersect][:,[2,1]], num_centorids = 3)
                        refine_prompt =  np.array(torch.stack(selected_points).cpu())
                        labels = np.array([1]*refine_prompt.shape[0], np.int32)
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_id,
                            obj_id=ann_obj_id,
                            points=refine_prompt,
                            labels=labels,
                        )

                    video_segments = {}  # video_segments contains the per-frame segmentation results
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    mask3d =[(0, torch.zeros((n_spp), dtype=torch.int8, device=self.device))]
                    ######### Perframe SPP lifting #########
                    for view_id in range(len(video_segments)):
                        mapping = mappings_video[view_id].to(self.device)
                        mask2ds = []
                        # Track ID + 2D mask
                        for key_of_mask in video_segments[view_id].keys():
                            mask2ds.append((key_of_mask, torch.tensor(video_segments[view_id][key_of_mask][0]).to(self.device)))

                        total_spp_points = torch_scatter.scatter((mapping[:, 3] == 1).float(), spp, dim=0, reduce="sum")
                        spp_weights = torch.zeros((n_spp), dtype=torch.float32, device=self.device)
                        idx = torch.nonzero(mapping[:, 3] == 1).view(-1)
                        
                        # Granularites
                        for (track_id, mask2d) in mask2ds:
                            highlight_points = idx[
                                mask2d[mapping[idx][:, [1, 2]][:, 0], mapping[idx][:, [1, 2]][:, 1]].nonzero(as_tuple=True)[0]
                            ].long()

                            sieve_mask = torch.zeros((self.n_points), device=self.device)
                            sieve_mask[highlight_points] = 1

                            num_related_points = torch_scatter.scatter(sieve_mask.float(), spp, dim=0, reduce="sum")

                            spp_weights = torch.where(
                                total_spp_points==0, 0, num_related_points / total_spp_points
                            )
                            target_spp = torch.nonzero(spp_weights >= self.cfg.proposal3d.sppweight).view(-1)

                            if len(target_spp) < 1:
                                mask3d.append((track_id, torch.zeros((n_spp), dtype=torch.int8, device=self.device)))
                                continue
                            elif len(target_spp) == 1:
                                group_tmp = torch.zeros((n_spp), dtype=torch.int8, device=self.device)
                                group_tmp[target_spp] = 1
                                mask3d.append((track_id, group_tmp))
                            else:
                                pairwise_dc_dist = dc_feature_matrix[target_spp, :][:, target_spp]
                                pairwise_dc_dist[torch.eye((len(target_spp)), dtype=torch.bool, device=dc_feature_matrix.device)] = -10
                                max_dc_dist = torch.max(pairwise_dc_dist, dim=1)[0]
                                valid_spp = max_dc_dist >= 0.5

                                if valid_spp.sum() > 0:
                                    target_spp = target_spp[valid_spp]
                                    group_tmp = torch.zeros((n_spp), dtype=torch.int8, device=self.device)
                                    group_tmp[target_spp] = 1
                                    mask3d.append((track_id, group_tmp))
                                else:
                                    mask3d.append((track_id, torch.zeros((n_spp), dtype=torch.int8, device=self.device)))

                                            

                    ######### Visualize 2D mask of the video #########
                    if True:
                        vis_frame_stride = 1
                        plt.close("all")
                        for out_frame_idx in range(0, len(images), vis_frame_stride):
                            plt.figure(figsize=(6, 4))
                            plt.title(f"frame {out_frame_idx}")
                            plt.imshow(images[out_frame_idx])
                            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                                show_mask_video(out_mask, plt.gca(), obj_id=out_obj_id)
                            plt.savefig(
                            os.path.join("../debug/" + 'viz/' + str(out_frame_idx) + ".jpg"),
                            bbox_inches="tight",
                            dpi=300,
                            pad_inches=0.0,
                            )
                    return mask3d, video_segments

                mask3d_1, video_masks1 = forward_sam2(images1, mapping1, video_path = None)
                mask3d_2 = []
                video_masks2 = []
                # mask3d_2, video_masks2 = forward_sam2(images2, mapping2, video_path = None)

                #NOTE: Form a unified Cluster from clusters generated by the 2 videos
                def unify_cluster_OR(mask3d_video1, mask3d_video2):
                    # OR-ing every cluster into one cluster
                    # only considering the same ID
                    max_id = 0
                    for i in range (len(mask3d_video1)):
                        max_id = max(max_id, mask3d_video1[i][0])
                    for i in range (len(mask3d_video2)):
                        max_id = max(max_id, mask3d_video2[i][0])
                    
                    masks_granularity = torch.zeros((max_id + 1, n_spp),dtype = torch.int8 ,device = self.device)
                    for i in range (len(mask3d_video1)):
                        track_id, vect = mask3d_video1[i]
                        masks_granularity[track_id] = torch.logical_or(masks_granularity[track_id], vect)
                    for i in range (len(mask3d_video2)):
                        track_id, vect = mask3d_video2[i]
                        masks_granularity[track_id] = torch.logical_or(masks_granularity[track_id], vect)

                    return masks_granularity
                

                #NOTE: By Convex Optimization
                def unify_cluster_LP(mask3d_video1, mask3d_video2, video_segments1, video_segments2, video_mappings1, video_mappings2):
                    # Local Search
                    n_state = 0
                    video_metadata = []
                    mask3d = []
                    for view_id in range(len(video_segments1)):
                        mapping = video_mappings1[view_id].to(self.device)
                        mask2ds = []
                        # Track ID + 2D mask
                        for key_of_mask in video_segments1[view_id].keys():
                            mask2ds.append(torch.tensor(video_segments1[view_id][key_of_mask][0]).to(self.device))
                        mask2ds = torch.stack(mask2ds, dim = 0)
                        video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                        n_state+=1         
                    for view_id in range(len(video_segments2)):
                        mapping = video_mappings2[view_id].to(self.device)
                        mask2ds = []
                        # Track ID + 2D mask
                        for key_of_mask in video_segments2[view_id].keys():
                            mask2ds.append(torch.tensor(video_segments2[view_id][key_of_mask][0]).to(self.device))
                        mask2ds = torch.stack(mask2ds, dim = 0)
                        video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                        n_state+=1
                    for mk in mask3d_video1:
                        if mk[1].sum().item() != 0:
                            mask3d.append(mk[1])
                    for mk in mask3d_video2:
                        if mk[1].sum().item() != 0:
                            mask3d.append(mk[1])    
                    
                    # Empty proposals 
                    if len(mask3d) == 0 or len(video_metadata) == 0:
                        return torch.zeros((1,1))

                    mask3ds = torch.stack(mask3d, dim = 0)
                    mask3ds_spp = mask3ds.clone() # sppwise
                    mask3ds = mask3ds[:,self.spp] # pointwise

                    ### Choosing topk views
                    loss_value = []
                    for (mask_id, mk) in enumerate(mask3ds):
                        _, _, _, score = cost_LP(mask3ds[mask_id], video_metadata)
                        loss_value.append(score)
                    loss_value = torch.tensor(loss_value)
                    _, selected_views = torch.topk(loss_value, k = min(cfg.proposal3d.view_optim,len(mask3d)), largest = False)
                    mask3ds = mask3ds[selected_views]
                    mask3ds_spp = mask3ds_spp[selected_views]
                    video_metadata = [video_metadata[vv] for vv in selected_views]

                    ### Bruteforce (Worst)
                    loss = 1e18
                    target = None
                    for num in range(1, 1<<mask3ds.shape[0], 1):
                        vert = []
                        for bit in range(10):
                            if (num & (1<<bit)) != 0:
                                vert.append(bit)
                        temp = torch.any(mask3ds[vert], dim = 0)
                        fg, bg, smask, score = cost_LP(temp, video_metadata)    
                        if score < loss:
                            loss = score
                            target = torch.any(mask3ds_spp[vert], dim = 0)
                    return target

                def unify_cluster_DP(mask3d_video1, mask3d_video2, video_segments1, video_segments2, video_mappings1, video_mappings2):
                    # Local Search
                    n_state = 0
                    video_metadata = []
                    mask3d = []
                    for view_id in range(len(video_segments1)):
                        mapping = video_mappings1[view_id].to(self.device)
                        mk = mask3d_video1[view_id][1]
                        if mk.sum().item() == 0:
                            continue
                        mask3d.append(mk)
                        mask2ds = []
                        # Track ID + 2D mask
                        for key_of_mask in video_segments1[view_id].keys():
                            mask2ds.append(torch.tensor(video_segments1[view_id][key_of_mask][0]).to(self.device))
                        mask2ds = torch.stack(mask2ds, dim = 0)
                        video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                        n_state+=1
                    
                    # Empty proposals 
                    if len(mask3d) == 0 or len(video_metadata) == 0:
                        return torch.zeros((1,1))

                    mask3ds = torch.stack(mask3d, dim = 0)
                    mask3ds_spp = mask3ds.clone() # sppwise
                    mask3ds = mask3ds[:,self.spp] # pointwise

                    ### Choosing greedy sort views
                    loss_value = []
                    for (mask_id, mk) in enumerate(mask3ds):
                        _, _, _, score = cost_LP(mask3ds[mask_id], video_metadata)
                        loss_value.append(score)
            
                    loss_key = sorted(range(len(loss_value)), key=lambda k: loss_value[k])
                    mask3ds = mask3ds[loss_key]
                    mask3ds_spp = mask3ds_spp[loss_key]
                    video_metadata = [video_metadata[vv] for vv in loss_key]

                    ### Dynamic Programming SUB-Opt
                    num_views = len(video_metadata)
                    rows, cols = (num_views, 2)
                    d = torch.zeros((rows, cols), dtype = torch.int64, device = self.device)
                    d[:,:] = 1e18
                    
                    _,_,_,d[0, 1] = cost_LP(mask3ds[0], video_metadata)
                    spp_set = {0: mask3ds[0]}

                    for i in range (1, num_views, 1):
                        d[i, 0] = min(d[i - 1, 0], d[i - 1, 1]).item()
                        d[i, 1] = cost_LP(torch.logical_or(spp_set[i - 1], mask3ds[i]), video_metadata)[3]
                        # trackback update
                        if d[i, 1] < d[i, 0]:
                            _,_,_,tmp = cost_LP(mask3ds[i], video_metadata)
                            if tmp < d[i, 1].item():
                                spp_set[i] = mask3ds[i]
                                d[i, 1] = tmp
                            else:
                                spp_set[i] = torch.logical_or(spp_set[i - 1], mask3ds[i])
                        else:
                            spp_set[i] = spp_set[i - 1]

                    return spp_set[num_views - 1]

                def unify_cluster_BIP(mask3d_video1, mask3d_video2, video_segments1, video_segments2, video_mappings1, video_mappings2):
                    # Local Search
                    n_state = 0
                    video_metadata = []
                    mask3d = []
                    for view_id in range(len(video_segments1)):
                        mapping = video_mappings1[view_id].to(self.device)
                        mask2ds = []
                        # Track ID + 2D mask
                        for key_of_mask in video_segments1[view_id].keys():
                            mask2ds.append(torch.tensor(video_segments1[view_id][key_of_mask][0]).to(self.device))
                        mask2ds = torch.stack(mask2ds, dim = 0)
                        video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                        n_state+=1         
                    for view_id in range(len(video_segments2)):
                        mapping = video_mappings2[view_id].to(self.device)
                        mask2ds = []
                        # Track ID + 2D mask
                        for key_of_mask in video_segments2[view_id].keys():
                            mask2ds.append(torch.tensor(video_segments2[view_id][key_of_mask][0]).to(self.device))
                        mask2ds = torch.stack(mask2ds, dim = 0)
                        video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                        n_state+=1
                    for mk in mask3d_video1:
                        if mk[1].sum().item() != 0:
                            mask3d.append(mk[1])
                    for mk in mask3d_video2:
                        if mk[1].sum().item() != 0:
                            mask3d.append(mk[1])    
                    
                    # Empty proposals 
                    if len(mask3d) == 0 or len(video_metadata) == 0:
                        return torch.zeros((1,1))

                    mask3ds = torch.stack(mask3d, dim = 0)
                    mask3ds_spp = mask3ds.clone() # sppwise
                    mask3ds = mask3ds[:,self.spp] # pointwise
                                        
                    mask3d = [vv.cpu().numpy() for vv in mask3d]
                    A = csr_matrix(mask3d)
                    T = csr_matrix()
                    # ### Choosing topk views
                    loss_value = []
                    for (mask_id, mk) in enumerate(mask3ds):
                        _, _, _, score = cost_LP(mask3ds[mask_id], video_metadata)
                        loss_value.append(score)
                    B = np.array(loss_value)
                    # Define the number of variables (columns of A)
                    n_vars = A.shape[1]
                    # Create a BIP problem instance
                    problem = pulp.LpProblem("Binary_Optimization", pulp.LpMinimize)  # Change to LpMaximize
                    # Define the binary decision variables X
                    X = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_vars)]
                    # Objective function: Maximize sum(AX)
                    objective = pulp.lpMin(pulp.lpSum(A[i, j] * X[j] for j in range(n_vars)))
                    problem += objective, "Maximize_Sum_AX"
                    # Add the constraint AX = B
                    for i in range(A.shape[0]):
                        problem += pulp.lpSum(A[i, j] * X[j] for j in range(n_vars)) == B[i], f"constraint_{i}"
                    problem.solve()
                    
                    target = torch.zeros((A.shape[1]),  dtype=torch.int8, device=self.device)
                    for X in problem.variables():
                        target[int(X.name.replace('x_', ''))] = 1.0
                    return target
                    # loss_value = torch.tensor(loss_value)
                    # _, selected_views = torch.topk(loss_value, k = min(cfg.proposal3d.view_optim,len(mask3d)), largest = False)
                    # mask3ds = mask3ds[selected_views]
                    # mask3ds_spp = mask3ds_spp[selected_views]
                    # video_metadata = [video_metadata[vv] for vv in selected_views]

                    # ### Bruteforce (Worst)
                    # loss = 1e18
                    # target = None
                    # for num in range(1, 1<<mask3ds.shape[0], 1):
                    #     vert = []
                    #     for bit in range(10):
                    #         if (num & (1<<bit)) != 0:
                    #             vert.append(bit)
                    #     temp = torch.any(mask3ds[vert], dim = 0)
                    #     fg, bg, smask, score = cost_LP(temp, video_metadata)    
                    #     if score < loss:
                    #         loss = score
                    #         target = torch.any(mask3ds_spp[vert], dim = 0)
                    # return target

                # proposals_from_spp = unify_cluster_OR(mask3d_1, mask3d_2)
                proposals_from_spp = unify_cluster_DP(mask3d_1, mask3d_2, video_masks1, video_masks2, mapping1, mapping2).unsqueeze(0)
                for proposal_from_spp in proposals_from_spp:
                    # Consider each granularity
                    if proposal_from_spp.sum() == 0: # empty mask
                        continue
                    ######### Decide whether to add this mask to proposal bank #########
                    # NOTE: (We don't use this anymore) should we consider whether to add (or just randomly throw it to the bank)?
                    proposal_bank.append(proposal_from_spp)
                    mask_flag = True

            if mask_flag == False: # Empy GT target -- No 3D proposal
                print('Empty proposal')
                proposal_bank.append(torch.zeros((self.n_points), dtype=torch.int8, device=self.device))


        proposal_bank = torch.stack(proposal_bank)
        if proposal_bank.shape[1] != self.n_points:
            proposals_pred = proposal_bank[:, spp]  # .bool()
        else:
            proposals_pred = proposal_bank
        
        gt_proposals = torch.stack(gt_proposals).to(self.device)

        ### Cannot Found 3D instance - discard 
        gt_proposals = gt_proposals[(proposals_pred.sum(-1)!=0)]
        proposals_pred = proposals_pred[(proposals_pred.sum(-1)!=0)]

        inter = torch.logical_and(proposals_pred, gt_proposals[:]).sum(-1)
        uni = torch.logical_or(proposals_pred, gt_proposals[:]).sum(-1)
        return proposals_pred.cpu(), inter, uni