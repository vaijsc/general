import os
import cv2
import pickle
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
from segment_anything_hq import sam_model_registry, SamPredictor


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
)
from torchmetrics.functional import pairwise_cosine_similarity
from util3d.pointnet2.pointnet2_utils import furthest_point_sample, ball_query
from random import randrange
from scipy.spatial import KDTree

from pyqubo import Array, Constraint, Placeholder
# import shrdr
from dimod import BinaryQuadraticModel
import dimod
import neal
import random

from qubobrute import *
from numba import cuda
import torch
import concurrent.futures
import multiprocessing as mp

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def search_subset(start, end, mask3ds, mask3ds_spp, video_metadata):
    local_loss = 1e18
    local_target = None
    for num in range(start, end):
        vert = []
        for bit in range(10):
            if (num & (1 << bit)) != 0:
                vert.append(bit)
        temp = torch.any(mask3ds[vert], dim=0)
        fg, bg, smask, score = cost_LP(temp, video_metadata)
        if score < local_loss:
            local_loss = score
            local_target = torch.any(mask3ds_spp[vert], dim=0)
    return local_loss, local_target

def multithreaded_bruteforce(mask3ds, mask3ds_spp, video_metadata, num_threads=4):
    max_num = 1 << mask3ds.shape[0]
    chunk_size = max_num // num_threads

    futures = []
    best_loss = 1e18
    best_target = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != num_threads - 1 else max_num
            futures.append(executor.submit(search_subset, start, end, mask3ds, mask3ds_spp, video_metadata))

        for future in concurrent.futures.as_completed(futures):
            loss, target = future.result()
            if loss < best_loss:
                best_loss = loss
                best_target = target

    return best_loss, best_target

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


class SAM_L2:
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
        
        self.n_object = 0
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

            dic = {"mapping": mapping.cpu(), "image": rgb_img}
            self.pcd_list.append(dic)

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
        mask_bank = []
        id_bank = []
        obj_bank = []
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
        
        
        # Iterative FPS
        visited_spp = torch.ones((n_spp), dtype = torch.bool, device = self.device)
        previous = -1
        END_FLAG = False

        while END_FLAG == False:
            leftover = visited_spp.sum().item() # how many spp left?
            # if previous == leftover: # break out as it converged
            #     break
            if mean_spp[:,0].sum(dim=-1).item() == 0:
                break
            previous = visited_spp.sum().item() 
            ################################################################
            print('---------- Remaining ', str(leftover), ' spps ---------')
            # Furthest Point Sampling
            N_Centorids = self.cfg.proposal3d.ncluster_fps
            # N_Centorids = int(n_spp/4)
            if visited_spp.sum().item() < N_Centorids:
                spp_ids = torch.where(visited_spp == True)[0]
                END_FLAG = True
            else:
                mean_spp[visited_spp == False, 0] = 0
                mean_spp[visited_spp == False, 1] = 0
                mean_spp[visited_spp == False, 2] = 0
                fps_inds = furthest_point_sample(mean_spp.to(torch.float).unsqueeze(0), N_Centorids).long()[0]
                spp_ids = np.array(fps_inds.cpu())

            # For each spp
            for spp_id in tqdm(spp_ids):
                ###NOTE: Fill visited conditions & better visualization & fewer masks

                # mean_spp[visited_spp == False, 0] = 0
                # mean_spp[visited_spp == False, 1] = 0
                # mean_spp[visited_spp == False, 2] = 0

                ###NOTE: Anchor conditions
                if mean_spp[spp_id, 0] == 0 and mean_spp[spp_id, 1] == 0 and mean_spp[spp_id, 2] == 0:
                    continue
                else:
                    mean_spp[spp_id, 0] = 0
                    mean_spp[spp_id, 1] = 0
                    mean_spp[spp_id, 2] = 0

                mask = (spp == spp_id)
                # NOTE: Calculate pixel' distribution on every frame given a spp  
                # shape =  (nFrame)  
                pixel_distribution = point_distribution[:,spp_id]
                # vals, peaks = torch.topk(pixel_distribution, k = 1) # Old IDEA

                # NOTE: for future dev ideas: what if max == npoint in spp (and there are more than 2 peaks?)
                ### K Neareast spherical Neighbors Spherical
                k_closest_spp = self.spherical_query(mean_spp_origin[spp_id], mean_spp_origin, neighbors = cfg.proposal3d.neighbors)

                ### K Neareast spherical Neighbors KDTree                
                # k_closest_spp = self.kdtree_query(mean_spp_origin[spp_id], mean_spp_origin, neighbors=cfg.proposal3d.neighbors)

                ####### NOTE: weighted distribution #######
                weighted_distribution = torch.mean((point_distribution[:,k_closest_spp]/sieve_of_spp[k_closest_spp]), dim = -1)
                weighted_distribution *= pixel_distribution

                ####Ablation study swapping pixel_dis, weigthed_dis
                # weighted_distribution = pixel_distribution.clone()
                ####
                
                vals, peaks = torch.topk(weighted_distribution, k = 1)
                
                # Find superpoint interval
                for peak_id in peaks:
                    peak_id = peak_id.item()
                    peak_value = pixel_distribution[peak_id].item()
                    if peak_value == 0:
                        continue
                    
                    ###### NOTE: Old version, track from first frame, How can we leverage the reprompting technique in SAM-2
                    # left = peak_id
                    # right = peak_id
                    # while(left > 0 and pixel_distribution[left - 1] * self.cfg.proposal3d.video_factor > peak_value):
                    #     left -= 1
                    # while(right < pixel_distribution.shape[0] - 1 and pixel_distribution[right + 1] * self.cfg.proposal3d.video_factor > peak_value):
                    #     right += 1
                    ####### NOTE: track the whole video, time consuming
                    left = 0
                    right = pixel_distribution.shape[0] - 1


                    # Backward Video 
                    numback = len(torch.where(weighted_distribution[left:peak_id + 1] > 0.0)[0])
                    target_backward = (peak_id - torch.topk(weighted_distribution[left:peak_id + 1], k = min(self.cfg.proposal3d.weighted_views, numback), largest = True)[1]).tolist()
                    target_backward.sort()
                    
                    images1, mapping1 = self.construct_video(self.pcd_list[left:peak_id + 1][::-1], target = target_backward)
                    image_id1 = (peak_id - np.array(target_backward)).tolist()
                    # Forward Video
                    numforward = len(torch.where(weighted_distribution[peak_id:right + 1] > 0.0)[0])
                    target_forward = (torch.topk(weighted_distribution[peak_id:right + 1], k = min(self.cfg.proposal3d.weighted_views, numforward), largest = True)[1]).tolist()
                    
                    target_forward.sort()
                    images2, mapping2 = self.construct_video(self.pcd_list[peak_id: right + 1], target = target_forward)
                    image_id2 = (np.array(target_forward) + peak_id).tolist()

                    # Concat 2 videos
                    image_id1 = image_id1 + image_id2
                    images1 = images1 + images2
                    mapping1 = mapping1 + mapping2                  

                    # Prompt Precomputation
                    mapping = self.pcd_list[peak_id]['mapping'].to(self.device)
                    intersect = torch.logical_and(mask, mapping[:,3])
                    # 1: point prompt granularities || 3: highest score prompts
                    start_image = images1[0]
                    start_image = np.array(start_image.convert("RGB"))
                    self.image_predictor.set_image(start_image)
                    RANSAC_Rounds = 1
                    maximum_score = 0
                    init_mask_prompts = None
                    while RANSAC_Rounds > 0:
                        # NOTE: PFS sample
                        selected_points = furthest_sampling2d(mapping[intersect][:,[2,1]], num_centorids = 3)
                        points_prompt =  np.array(torch.stack(selected_points).cpu())

                        # NOTE: random sample point 
                        # nums_mapping_point = mapping[intersect][:,[2,1]].shape[0]
                        # selected_points = [mapping[intersect][:,[2,1]][torch.randint(0, nums_mapping_point, (3,))]]
                        # points_prompt =  np.array(torch.stack(selected_points).cpu()).squeeze(0)
                        
                        labels = np.array([1]*points_prompt.shape[0], np.int32)
                        masks_prompts, scores, logits = self.image_predictor.predict(
                            point_coords=points_prompt,
                            point_labels=labels,
                            multimask_output=False,
                        )
                        if scores[0] > maximum_score:
                            init_mask_prompts = masks_prompts
                            maximum_score = scores[0]
                        RANSAC_Rounds -= 1

                    # The anchored mask has to be good enough
                    if maximum_score < 0.6:
                        continue

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
                            # mask prompt first frame
                            _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                                inference_state=inference_state,
                                frame_idx=ann_frame_idx,
                                obj_id=ann_obj_id,
                                mask=mk_prompt,
                            )

                        ###################### Point 3D aware refining Prompt ######################
                        for frame_id in range(1, len(mappings_video), max(1, int(len(mappings_video)/10))):
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
                        if False:
                            vis_frame_stride = 1
                            plt.close("all")
                            for out_frame_idx in range(0, len(images), vis_frame_stride):
                                plt.figure(figsize=(6, 4))
                                plt.title(f"frame {out_frame_idx}")
                                plt.imshow(images[out_frame_idx])
                                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                                    show_mask_video(out_mask, plt.gca(), obj_id=out_obj_id)
                                plt.savefig(
                                os.path.join("../debug/" + scene_id + "/sam2/" + str(out_frame_idx) + ".jpg"),
                                bbox_inches="tight",
                                dpi=300,
                                pad_inches=0.0,
                                )
                        return mask3d, video_segments

                    mask3d_1, video_masks1 = forward_sam2(images1, mapping1, video_path = None)
                    storage = {'mask3d': mask3d_1, 'video_mask': video_masks1, 'image_id': image_id1}
                    
                    # Dumping storage
                    # save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output, scene_id)
                    # os.makedirs(save_dir_cluster, exist_ok=True)
                    # with open(os.path.join(save_dir_cluster, '{0:05}'.format(self.n_object) + '.pth'), "wb") as fp:
                    #     pickle.dump(storage, fp)
                    # storage = None
                    # del storage
                    # torch.cuda.empty_cache()

                    # self.n_object += 1

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

                        ## Choosing topk views
                        loss_value = []
                        for (mask_id, mk) in enumerate(mask3ds):
                            _, _, _, score = cost_LP(mask3ds[mask_id], video_metadata)
                            loss_value.append(score)
                        loss_value = torch.tensor(loss_value)
                        _, selected_views = torch.topk(loss_value, k = min(cfg.proposal3d.view_optim,len(mask3d)), largest = False)
                        mask3ds = mask3ds[selected_views]
                        mask3ds_spp = mask3ds_spp[selected_views]
                        video_metadata = [video_metadata[vv] for vv in selected_views]

                        # ### Bruteforce (Worst)
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
                        # best_loss, target = multithreaded_bruteforce(mask3ds, mask3ds_spp, video_metadata, num_threads=32)
                        return target

                    def unify_cluster_DP(mask3d_video1, video_segments1, video_mappings1, image_id1, cfg, scene_id):
                        # Local Search
                        n_state = 0
                        video_metadata = []
                        mask3d = []
                        mask2d_save = []
                        im_mask_id = []
                        for view_id in range(len(video_segments1)):
                            mapping = video_mappings1[view_id]
                            mk = mask3d_video1[view_id][1]
                            if mk.sum().item() == 0:
                                continue
                            mask3d.append(mk)
                            mask2ds = []
                            # Track ID + 2D mask
                            for key_of_mask in video_segments1[view_id].keys():
                                mask2ds.append(torch.tensor(video_segments1[view_id][key_of_mask][0]).to(self.device))
                            mask2ds = torch.stack(mask2ds, dim = 0)
                            mask2d_save.append(mask2ds)
                            im_mask_id.append(image_id1[view_id])
                            video_metadata.append({'mapping': mapping, 'mask2d': mask2ds, 'image_id': image_id1[view_id]})
                            n_state+=1
                        
                        # Empty proposals 
                        if len(mask3d) == 0 or len(video_metadata) == 0:
                            return None, torch.zeros((1,1)), torch.zeros((1,1)), None
                        
                        #### SAVING DIR
                        storage = [{'mask2d': mask2d_save, 'im_mask2d_id': im_mask_id}]
                        save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output, scene_id)
                        os.makedirs(save_dir_cluster, exist_ok=True)
                        with open(os.path.join(save_dir_cluster, '{0:05}'.format(self.n_object) + '.pth'), "wb") as fp:
                            pickle.dump(storage, fp)
                        storage = None
                        del storage
                        torch.cuda.empty_cache()
                        self.n_object += 1
                        ####

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

                        ### Dynamic Programming SUB-Opt O(N)
                        num_views = len(video_metadata)
                        rows, cols = (num_views, 2)
                        d = torch.zeros((rows, cols), dtype = torch.int64, device = self.device)
                        d[:,:] = 1e18
                        
                        _,_,_,d[0, 1] = cost_LP(mask3ds[0], video_metadata)
                        
                        spp_set = {0: mask3ds[0]}
                        track_set = {0:[video_metadata[0]['image_id']]}
                        mask_set = {0:[video_metadata[0]['mask2d']]}

                        for i in range (1, num_views, 1):
                            d[i, 0] = min(d[i - 1, 0], d[i - 1, 1]).item()
                            d[i, 1] = cost_LP(torch.logical_or(spp_set[i - 1], mask3ds[i]), video_metadata)[3]
                            # trackback update
                            if d[i, 1] < d[i, 0]:
                                _,_,_,tmp = cost_LP(mask3ds[i], video_metadata)
                                if tmp < d[i, 1].item():
                                    spp_set[i] = mask3ds[i]
                                    track_set[i] = [video_metadata[i]['image_id']]
                                    mask_set[i] = [video_metadata[i]['mask2d']]
                                    d[i, 1] = tmp
                                else:
                                    spp_set[i] = torch.logical_or(spp_set[i - 1], mask3ds[i])
                                    track_set[i] = track_set[i - 1] + [video_metadata[i]['image_id']]
                                    mask_set[i] = mask_set[i - 1] + [video_metadata[i]['mask2d']]
                            else:
                                spp_set[i] = spp_set[i - 1]
                                track_set[i] = track_set[i - 1]
                                mask_set[i] = mask_set[i - 1]


                        return spp_set[num_views - 1], mask_set[num_views - 1], track_set[num_views - 1], self.n_object - 1

                    def unify_cluster_LP2(mask3d_video1, mask3d_video2, video_segments1, video_segments2, video_mappings1, video_mappings2):
                        n_state = 0
                        video_metadata = []
                        mask3d = []
                        
                        # Prepare video metadata for all views in video1
                        for view_id in range(len(video_segments1)):
                            mapping = video_mappings1[view_id].to(self.device)
                            mask2ds = []
                            for key_of_mask in video_segments1[view_id].keys():
                                mask2ds.append(torch.tensor(video_segments1[view_id][key_of_mask][0]).to(self.device))
                            mask2ds = torch.stack(mask2ds, dim=0)
                            video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                            n_state += 1
                        
                        # Prepare video metadata for all views in video2
                        for view_id in range(len(video_segments2)):
                            mapping = video_mappings2[view_id].to(self.device)
                            mask2ds = []
                            for key_of_mask in video_segments2[view_id].keys():
                                mask2ds.append(torch.tensor(video_segments2[view_id][key_of_mask][0]).to(self.device))
                            mask2ds = torch.stack(mask2ds, dim=0)
                            video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                            n_state += 1

                        # Collect valid 3D masks from both videos
                        for mk in mask3d_video1:
                            if mk[1].sum().item() != 0:
                                mask3d.append(mk[1])
                        for mk in mask3d_video2:
                            if mk[1].sum().item() != 0:
                                mask3d.append(mk[1])

                        # Return empty mask if no valid proposals are available
                        if len(mask3d) == 0 or len(video_metadata) == 0:
                            return torch.zeros((1, 1))

                        # Stack 3D masks and prepare pointwise masks
                        mask3ds = torch.stack(mask3d, dim=0)
                        mask3ds_spp = mask3ds.clone()  # sppwise
                        mask3ds = mask3ds[:, self.spp]  # pointwise

                        ### Choosing top k views based on loss value
                        loss_value = []
                        for mask_id in range(mask3ds.shape[0]):
                            _, _, _, score = cost_LP(mask3ds[mask_id], video_metadata)
                            loss_value.append(score)
                        
                        loss_value = torch.tensor(loss_value)
                        _, selected_views = torch.topk(loss_value, k=min(cfg.proposal3d.view_optim, len(mask3d)), largest=False)
                        mask3ds = mask3ds[selected_views]
                        mask3ds_spp = mask3ds_spp[selected_views]
                        video_metadata = [video_metadata[vv] for vv in selected_views]

                        ### QUBO Optimization using Simulated Annealing (Classical Solver)
                        # Number of masks to consider for QUBO
                        n_masks = mask3ds.shape[0]
                        
                        # Prepare the Binary Quadratic Model
                        bqm = BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

                        # Fill the QUBO matrix with diagonal and off-diagonal elements
                        for i in range(n_masks):
                            mask_i = mask3ds[i]
                            fg_i, bg_i, smask_i, score_i = cost_LP(mask_i, video_metadata)
                            bqm.add_variable(i, score_i)  # Diagonal element: cost of selecting mask i

                            for j in range(i+1, n_masks):
                                mask_j = mask3ds[j]
                                combined_score = cost_LP(mask_i + mask_j, video_metadata)[-1]  # Combined cost for i and j
                                interaction_cost = combined_score - (score_i + cost_LP(mask_j, video_metadata)[-1])  # Interaction term
                                bqm.add_interaction(i, j, interaction_cost)

                        # Use Simulated Annealing Sampler (Classical Solver) instead of D-Wave
                        sampler = dimod.SimulatedAnnealingSampler()
                        sampleset = sampler.sample(bqm, num_reads=100)

                        # Get the best solution from the sampleset
                        best_solution = sampleset.first.sample

                        
                        # Select the masks based on the best solution
                        selected_masks = []
                        for mask_id, selected in best_solution.items():
                            if selected == 1:
                                selected_masks.append(mask3ds_spp[mask_id])

                        # Return the final mask based on the selected views
                        if len(selected_masks) > 0:
                            target = torch.any(torch.stack(selected_masks, dim=0), dim=0)
                        else:
                            target = None
                        
                        return target

                    def unify_cluster_LP3(mask3d_video1, mask3d_video2, video_segments1, video_segments2, video_mappings1, video_mappings2):
                        n_state = 0
                        video_metadata = []
                        mask3d = []
                        
                        # Prepare video metadata for all views in video1
                        for view_id in range(len(video_segments1)):
                            mapping = video_mappings1[view_id].to(self.device)
                            mask2ds = []
                            for key_of_mask in video_segments1[view_id].keys():
                                mask2ds.append(torch.tensor(video_segments1[view_id][key_of_mask][0]).to(self.device))
                            mask2ds = torch.stack(mask2ds, dim=0)
                            video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                            n_state += 1
                        
                        # Prepare video metadata for all views in video2
                        for view_id in range(len(video_segments2)):
                            mapping = video_mappings2[view_id].to(self.device)
                            mask2ds = []
                            for key_of_mask in video_segments2[view_id].keys():
                                mask2ds.append(torch.tensor(video_segments2[view_id][key_of_mask][0]).to(self.device))
                            mask2ds = torch.stack(mask2ds, dim=0)
                            video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                            n_state += 1
                        
                        # Collect valid 3D masks from both videos
                        for mk in mask3d_video1:
                            if mk[1].sum().item() != 0:
                                mask3d.append(mk[1])
                        for mk in mask3d_video2:
                            if mk[1].sum().item() != 0:
                                mask3d.append(mk[1])

                        # Return empty mask if no valid proposals are available
                        if len(mask3d) == 0 or len(video_metadata) == 0:
                            return torch.zeros((1, 1))

                        # Stack 3D masks and prepare pointwise masks
                        mask3ds = torch.stack(mask3d, dim=0)
                        mask3ds_spp = mask3ds.clone()  # sppwise
                        mask3ds = mask3ds[:, self.spp]  # pointwise

                        ### Choosing top k views based on loss value
                        loss_value = []
                        for mask_id in range(mask3ds.shape[0]):
                            _, _, _, score = cost_LP(mask3ds[mask_id], video_metadata)
                            loss_value.append(score)

                        loss_value = torch.tensor(loss_value)
                        _, selected_views = torch.topk(loss_value, k=min(cfg.proposal3d.view_optim, len(mask3d)), largest=False)

                        mask3ds = mask3ds[selected_views]
                        mask3ds_spp = mask3ds_spp[selected_views]
                        video_metadata = [video_metadata[vv] for vv in selected_views]

                        ### QUBO Optimization using Simulated Annealing (Classical Solver)
                        n_masks = mask3ds.shape[0]
                        
                        bqm = BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

                        # Scale factor to balance individual and interaction costs
                        scale_factor = 1.0  # Adjust this value as needed

                        # Fill the QUBO matrix with diagonal and off-diagonal elements
                        for i in range(n_masks):
                            mask_i = mask3ds[i]
                            fg_i, bg_i, smask_i, score_i = cost_LP(mask_i, video_metadata)
                            bqm.add_variable(i, score_i)  # Diagonal element: cost of selecting mask i

                            for j in range(i+1, n_masks):
                                mask_j = mask3ds[j]
                                combined_mask = torch.logical_or(mask_i, mask_j)  # Use logical OR instead of addition
                                combined_score = cost_LP(combined_mask, video_metadata)[-1]
                                interaction_cost = scale_factor * (combined_score - score_i - cost_LP(mask_j, video_metadata)[-1])
                                bqm.add_interaction(i, j, interaction_cost)

                        # Use Simulated Annealing Sampler with custom parameters
                        sampler = dimod.SimulatedAnnealingSampler()
                        sampleset = sampler.sample(bqm, num_reads=100, num_sweeps=1000, beta_range=(0.1, 10))

                        # Get the best solution from the sampleset
                        best_solution = sampleset.first.sample
                        
                        # Select the masks based on the best solution
                        selected_masks = []
                        for mask_id, selected in best_solution.items():
                            if selected == 1:
                                selected_masks.append(mask3ds_spp[mask_id])

                        # Return the final mask based on the selected views
                        if len(selected_masks) > 0:
                            target = torch.any(torch.stack(selected_masks, dim=0), dim=0)
                        else:
                            target = None
                        
                        return target

                    def unify_cluster_LP4(mask3d_video1, mask3d_video2, video_segments1, video_segments2, video_mappings1, video_mappings2):
                        n_state = 0
                        video_metadata = []
                        mask3d = []
                        
                        # Prepare video metadata for all views in video1
                        for view_id in range(len(video_segments1)):
                            mapping = video_mappings1[view_id].to(self.device)
                            mask2ds = []
                            for key_of_mask in video_segments1[view_id].keys():
                                mask2ds.append(torch.tensor(video_segments1[view_id][key_of_mask][0]).to(self.device))
                            mask2ds = torch.stack(mask2ds, dim=0)
                            video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                            n_state += 1

                        # Prepare video metadata for all views in video2
                        for view_id in range(len(video_segments2)):
                            mapping = video_mappings2[view_id].to(self.device)
                            mask2ds = []
                            for key_of_mask in video_segments2[view_id].keys():
                                mask2ds.append(torch.tensor(video_segments2[view_id][key_of_mask][0]).to(self.device))
                            mask2ds = torch.stack(mask2ds, dim=0)
                            video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                            n_state += 1

                        # Collect valid 3D masks from both videos
                        for mk in mask3d_video1:
                            if mk[1].sum().item() != 0:
                                mask3d.append(mk[1])
                        for mk in mask3d_video2:
                            if mk[1].sum().item() != 0:
                                mask3d.append(mk[1])

                        # Return empty mask if no valid proposals are available
                        if len(mask3d) == 0 or len(video_metadata) == 0:
                            return torch.zeros((1, 1))

                        # Stack 3D masks and prepare pointwise masks
                        mask3ds = torch.stack(mask3d, dim=0)
                        mask3ds_spp = mask3ds.clone()  # sppwise
                        mask3ds = mask3ds[:, self.spp]  # pointwise

                        # ### Choosing topk views
                        # loss_value = []
                        # for (mask_id, mk) in enumerate(mask3ds):
                        #     _, _, _, score = cost_LP(mask3ds[mask_id], video_metadata)
                        #     loss_value.append(score)
                        # loss_value = torch.tensor(loss_value)
                        # _, selected_views = torch.topk(loss_value, k = min(cfg.proposal3d.view_optim,len(mask3d)), largest = False)
                        # mask3ds = mask3ds[selected_views]
                        # mask3ds_spp = mask3ds_spp[selected_views]
                        # video_metadata = [video_metadata[vv] for vv in selected_views]
                        

                        # QUBO Optimization using Simulated Annealing (Classical Solver)
                        # Number of masks to consider for QUBO
                        n_masks = mask3ds.shape[0]
                        
                        # Prepare the Binary Quadratic Model
                        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

                        # Fill the QUBO matrix with diagonal and off-diagonal elements
                        for i in range(n_masks):
                            mask_i = mask3ds[i]
                            fg_i, bg_i, smask_i, score_i = cost_LP(mask_i, video_metadata)
                            bqm.add_variable(i, score_i)  # Diagonal element: cost of selecting mask i

                            for j in range(i+1, n_masks):
                                mask_j = mask3ds[j]
                                combined_mask_or = torch.logical_or(mask_i, mask_j)
                                combined_score_or = cost_LP(combined_mask_or, video_metadata)[-1]  # Combined cost for i and j

                                combined_mask_and = torch.logical_and(mask_i, mask_j)
                                combined_score_and = cost_LP(combined_mask_and, video_metadata)[-1]  # Combined cost for i and j

                                score_j = cost_LP(mask_j, video_metadata)[-1]

                                # interaction_cost = combined_score_or - (score_i + score_j)  # Interaction term
                                # interaction_cost = combined_score_or
                                # interaction_cost = combined_score_and
                                # interaction_cost = combined_mask_or - combined_score_and # Interaction term A + B - AB

                                # breakpoint()
                                # interaction_cost = (score_i + score_j) - combined_score_or 
                                # interaction_cost = (score_i + score_j) - combined_score_and 
                                # interaction_cost = combined_score_and + (score_i + score_j)
                                # interaction_cost = (score_i + score_j)
                                
                                bqm.add_interaction(i, j, interaction_cost)

                        # Use Simulated Annealing Sampler (Classical Solver) instead of D-Wave


                        sampler = dimod.SimulatedAnnealingSampler()
                        sampleset = sampler.sample(bqm, num_reads=100, num_sweeps=1000, beta_range=(0.1, 10))
                        # sampleset = sampler.sample(bqm, num_reads=100)

                        # Get the best solution from the sampleset
                        best_solution = sampleset.first.sample
                        breakpoint()
                        
                        # Select the masks based on the best solution
                        selected_masks = []
                        for mask_id, selected in best_solution.items():
                            if selected == 1:
                                selected_masks.append(mask3ds_spp[mask_id])

                        # Return the final mask based on the selected views
                        if len(selected_masks) > 0:
                            target = torch.any(torch.stack(selected_masks, dim=0), dim=0)
                        else:
                            target = None
                        return target

                    def unify_cluster_LP5(mask3d_video1, mask3d_video2, video_segments1, video_segments2, video_mappings1, video_mappings2):
                        n_state = 0
                        video_metadata = []
                        mask3d = []
                        
                        # Prepare video metadata for all views in video1
                        for view_id in range(len(video_segments1)):
                            mapping = video_mappings1[view_id].to(self.device)
                            mask2ds = []
                            for key_of_mask in video_segments1[view_id].keys():
                                mask2ds.append(torch.tensor(video_segments1[view_id][key_of_mask][0]).to(self.device))
                            mask2ds = torch.stack(mask2ds, dim=0)
                            video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                            n_state += 1
                        
                        # Prepare video metadata for all views in video2
                        for view_id in range(len(video_segments2)):
                            mapping = video_mappings2[view_id].to(self.device)
                            mask2ds = []
                            for key_of_mask in video_segments2[view_id].keys():
                                mask2ds.append(torch.tensor(video_segments2[view_id][key_of_mask][0]).to(self.device))
                            mask2ds = torch.stack(mask2ds, dim=0)
                            video_metadata.append({'mapping': mapping, 'mask2d': mask2ds})
                            n_state += 1

                        # Collect valid 3D masks from both videos
                        for mk in mask3d_video1:
                            if mk[1].sum().item() != 0:
                                mask3d.append(mk[1])
                        for mk in mask3d_video2:
                            if mk[1].sum().item() != 0:
                                mask3d.append(mk[1])

                        # Return empty mask if no valid proposals are available
                        if len(mask3d) == 0 or len(video_metadata) == 0:
                            return torch.zeros((1, 1))

                        # Stack 3D masks and prepare pointwise masks
                        mask3ds = torch.stack(mask3d, dim=0)
                        mask3ds_spp = mask3ds.clone()  # sppwise
                        mask3ds = mask3ds[:, self.spp]  # pointwise
                        

                        # QUBO Optimization using Simulated Annealing (Classical Solver)
                        # Number of masks to consider for QUBO
                        n_masks = mask3ds.shape[0]
                        
                        # Create a binary array for the QUBO
                        x = Array.create('x', shape=(n_masks,), vartype='BINARY')

                        # Define the Hamiltonian using PyQUBO
                        H = 0
                        for i in range(n_masks):
                            mask_i = mask3ds[i]
                            fg_i, bg_i, smask_i, score_i = cost_LP(mask_i, video_metadata)
                            H += score_i * x[i]  # Diagonal term: cost of selecting mask i

                            for j in range(i+1, n_masks):
                                mask_j = mask3ds[j]
                                combined_mask = torch.logical_or(mask_i, mask_j)
                                combined_score = cost_LP(combined_mask, video_metadata)[-1]  # Combined cost for i and j
                                H += combined_score * x[i] * x[j]  # Interaction term for selecting masks i and j together

                        # Compile the model to a QUBO problem
                        model = H.compile()
                        qubo, offset = model.to_qubo(index_label=True)

                        # Convert the QUBO to matrix format compatible with QUBOBrute
                        q = to_mat(qubo)

                        # Run GPU-accelerated simulated annealing
                        energies, solutions = simulate_annealing_gpu(q, offset, n_iter=1000, n_samples=100_000, temperature=1.0, cooling_rate=0.99)

                        # Get the best solution from the results
                        best_solution = solutions[0]  # The first solution corresponds to the lowest energy state

                        # Select the masks based on the best solution
                        selected_masks = []
                        for mask_id, selected in enumerate(best_solution):
                            if selected == 1:
                                selected_masks.append(mask3ds_spp[mask_id])

                        # Return the final mask based on the selected views
                        if len(selected_masks) > 0:
                            target = torch.any(torch.stack(selected_masks, dim=0), dim=0)
                        else:
                            target = None
                        
                        return target

                    proposals_from_spp, corresponding_mask2d, corresponding_imageid, obj_id = unify_cluster_DP(mask3d_1, video_masks1, mapping1, image_id1, cfg, scene_id)
                    if proposals_from_spp != None:
                        proposals_from_spp = proposals_from_spp.unsqueeze(0)

                        if proposals_from_spp.shape[1] == self.n_points: # Point -> SPP
                            proposals_from_spp = custom_scatter_mean(
                                proposals_from_spp,
                                spp[None, :].expand(len(proposals_from_spp), -1),
                                dim=-1,
                                pool=True,
                                output_type=torch.float64,
                            )
                            proposals_from_spp = (proposals_from_spp >= 0.5)
                        for proposal_from_spp in proposals_from_spp:
                            # Consider each granularity
                            if proposal_from_spp.sum() == 0: # empty mask
                                continue
                            ######### Decide whether to add this mask to proposal bank #########
                            # NOTE: should we consider whether to add (or just randomly throw it to the bank)?
                            visited_spp[proposal_from_spp] = False
                            id_bank.append(corresponding_imageid)
                            mask_bank.append(corresponding_mask2d)
                            proposal_bank.append(proposal_from_spp)
                            obj_bank.append(obj_id)
        # Return results
        proposal_bank = torch.stack(proposal_bank).cpu()
        proposals_pred = proposal_bank[:, spp.cpu()]  # .bool()
        return proposals_pred.cpu(), mask_bank, id_bank, obj_bank