import os
import cv2
import numpy as np
import open3d as o3d
import pycocotools
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
# from detectron2.structures import BitMasks
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
from OmniScientModel.utils import prepare_osm, prepare_instruction
import torchvision.transforms as T


INPUT_SIZE = 1120
CONTEXT_ENLARGE_RATIO = 0.5
# The float16 affects final results slightly. Results in paper are obtained with float32.
TEST_DTYPE = torch.float16

def get_context_mask(mask, input_size, enlarge_ratio=0.5):
    if mask.sum() == 0:
        mask[0, 0, 0, 0] = True
        #raise ValueError("Got an empty mask!")
    
    if enlarge_ratio < 0:
        return torch.ones_like(mask).view(1, input_size, input_size)

    mask = mask.view(input_size, input_size)
    rows, cols = torch.where(mask)
    min_row, min_col = rows.min().item(), cols.min().item()
    max_row, max_col = rows.max().item(), cols.max().item()

    row_size = max_row - min_row + 1
    col_size = max_col - min_col + 1
    min_row = max(0, int(min_row - row_size * enlarge_ratio))
    max_row = min(input_size-1, int(max_row + row_size * enlarge_ratio))
    min_col = max(0, int(min_col - col_size * enlarge_ratio))
    max_col = min(input_size-1, int(max_col + col_size * enlarge_ratio))
    context_mask = torch.zeros_like(mask)
    context_mask[min_row:max_row+1, min_col:max_col+1] = 1
    return context_mask.view(1, input_size, input_size)


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


class OSM():
    def __init__(self, osm_checkpoint='/root/3dllm/weights/osm_final.pt'):
        # Prepare OSM model and instructions
        self.class_generator, self.processor = prepare_osm(osm_checkpoint=osm_checkpoint)
        self.lang_x, self.qformer_lang_x = prepare_instruction(
            self.processor, "What is in the segmentation mask? Assistant:")
        self.input_size = self.processor.image_processor.size["height"]
        self.track = 0

    @torch.no_grad()
    def process_multiple_image(self, images, seg_masks, img_id, interval, imgs, highlight_masks):
        ## retrieve qformer embed ->  aggregate into pc feature

        image_for_osms = []
        for im in images:
            image = Image.fromarray(im).convert("RGB")
            if min(image.size) == max(image.size):
                image = T.functional.resize(image, size=self.input_size, interpolation=T.functional.InterpolationMode.BICUBIC)
            else: # in our case, always this case
                image = T.functional.resize(image, size=self.input_size - 1, max_size=self.input_size, interpolation=T.functional.InterpolationMode.BICUBIC)
        
            image_for_seg = np.array(image)
            # pad to input_size x input_size
            padded_image = np.zeros(shape=(self.input_size, self.input_size, 3), dtype=np.uint8)
            padded_image[:image_for_seg.shape[0], :image_for_seg.shape[1]] = image_for_seg
            image_for_osm = Image.fromarray(padded_image)  

            image_for_osms.append(image_for_osm)
        
        # batching the masks for not being OOM, we choose batch mask = 100
        segmentation_masks = []
        for batch_id in trange(0, len(seg_masks), 100):
            ### Generating class
            batch_end = min(batch_id + 100, len(seg_masks))
            segmentation = torch.nn.functional.interpolate(torch.stack(seg_masks[batch_id:batch_end]).to(torch.float16).unsqueeze(1).cuda(),size=(image.size[1], image.size[0]), mode='bicubic')
            segmentation = (segmentation > 0.5) # booling the mask
            segmentation = segmentation.squeeze(1).cpu() 
            segmentation_masks.append(segmentation)
            torch.cuda.empty_cache()

        # batch initialization
        images_batch = []
        segmentation_masks_batch = []
        context_mask_batch = []
        qformer_input_ids_batch = []
        qformer_attention_mask_batch = []
        input_ids_batch = []
        attention_mask_batch = []
        
        pred_class = []
        class_probs = []

        batch_size = 50
        number_imgs = len(image_for_osms)
        input_size = self.processor.image_processor.size["height"]
        # [N, 3, inpsize, inpsize] ~~
        torch.cuda.empty_cache()
          
        image = self.processor(images=image_for_osms, return_tensors="pt")["pixel_values"].view(number_imgs, 3, input_size, input_size)

        num_mask = 0
        for segs in tqdm(segmentation_masks):
            for seg in segs:
                binary_mask = seg
                padded_binary_mask = np.zeros(shape=(input_size, input_size), dtype=np.uint8)
                padded_binary_mask[:binary_mask.shape[0], :binary_mask.shape[1]] = binary_mask
                binary_mask = padded_binary_mask
                binary_mask = torch.from_numpy(np.ascontiguousarray(binary_mask.copy().reshape(1, input_size, input_size)))

                # if binary_mask.sum() < 100:
                #     pred_class.append("")
                #     class_probs.append(0)
                #     continue
                
                binary_mask = binary_mask.view(1, 1, input_size, input_size).float()
                
                images_batch.append(image[img_id[num_mask]])
                segmentation_masks_batch.append(binary_mask)
                context_mask_batch.append(get_context_mask(binary_mask, input_size, 0.5).view(
                    1, 1, input_size, input_size))
                qformer_input_ids_batch.append(self.qformer_lang_x["input_ids"])
                qformer_attention_mask_batch.append(self.qformer_lang_x["attention_mask"])
                input_ids_batch.append(self.lang_x["input_ids"])
                attention_mask_batch.append(self.lang_x["attention_mask"])
                num_mask += 1
        '''
        MULTI version: having batch size
        batch of image and mask  (1 image, n mask)
        duplicate into n(1 image, 1 mask) per forwarding

        img_id = [nMask], keeping track mask belong to which image
        '''
        qformer_embed = []
        print('Querying OSM...')
        for batch_id in trange(0, len(images_batch), batch_size):
            ### Generating class
            batch_end = min(batch_id + batch_size, len(images_batch))
            with torch.no_grad():
                generated_output_qformer = self.class_generator.generate_1n_avg_qformer(
                    img_idd = img_id[batch_id:batch_end],
                    pixel_values=torch.stack(images_batch[batch_id:batch_end]).cuda().to(torch.float16),
                    qformer_input_ids=torch.cat(qformer_input_ids_batch[batch_id:batch_end]).cuda(),
                    qformer_attention_mask=torch.cat(qformer_attention_mask_batch[batch_id:batch_end]).cuda(),
                    input_ids=torch.cat(input_ids_batch[batch_id:batch_end]).cuda(),
                    attention_mask=torch.cat(attention_mask_batch[batch_id:batch_end]).cuda(),
                    cache_image_embeds=True,
                    segmentation_mask=torch.cat(segmentation_masks_batch[batch_id:batch_end]).cuda(),
                    input_context_mask=torch.cat(context_mask_batch[batch_id:batch_end]).cuda(),
                    dataset_type="any",
                    max_new_tokens=16,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True).cpu() # cpu for sake of mem
            for j in range(generated_output_qformer.shape[0]):
                qformer_embed.append(generated_output_qformer[j])
            torch.cuda.empty_cache()


        return qformer_embed
    @torch.no_grad()
    def query_llm(self, representatives):
        len_ = representatives.shape[0]
        # batch initialization
        input_ids_batch = []
        attention_mask_batch = []
        
        for _ in range(len_):
            input_ids_batch.append(self.lang_x["input_ids"])
            attention_mask_batch.append(self.lang_x["attention_mask"])        
        
        generated_output = self.class_generator.generate_1n_avg_language(
                                qformer_output=representatives.cuda(),
                                input_ids=torch.cat(input_ids_batch[0:1]).cuda(),
                                attention_mask=torch.cat(attention_mask_batch[0:1]).cuda(),
                                dataset_type="any",
                                max_new_tokens=16,
                                num_beams=1,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_text = generated_output["sequences"][0]
        try:
            gentext = self.processor.tokenizer.decode(generated_text).split('</s>')[1].strip()
            pred_class_tokenidx = self.processor.tokenizer.encode(gentext)
            scores = generated_output["scores"]

            scores = scores[:len(pred_class_tokenidx) -1] # minus one for bos token
            
            temp = 1.0
            probs = [torch.nn.functional.softmax(score / temp, dim=-1) for score in scores]
            pred_prob = 1.0
            for p_idx, prob in enumerate(probs):
                pred_idx = pred_class_tokenidx[p_idx + 1]
                pred_prob *= prob[0, pred_idx].cpu().item()
        except:
            gentext = 'None' # format of the sentence change due to avg
            pred_prob = 0
        return gentext, pred_prob

class FreeVocab_Reproduce:
    def __init__(self, cfg, class_names):
        # OpenAI CLIP
        self.device = "cuda:0"
        self.model = OSM(osm_checkpoint = '/root/3dllm/weights/osm_final.pt')


        self.cfg = cfg
        self.class_names = class_names
        # with torch.no_grad(), torch.cuda.amp.autocast():
        #     text_features = self.clip_model.clip_adapter.encode_text(clip.tokenize(class_names).to(self.device))
        #     text_features /= text_features.norm(dim=-1, keepdim=True)

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
                # Save Memory ScanNet200
                # rgb_img = cv2.resize(rgb_img,(depth.shape[1], depth.shape[0]))

            else:
                raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")
            dic = {"mapping": mapping.cpu(), "image": rgb_img, 'frame_id': frame_id}
            self.pcd_list.append(dic) 

    def align_majority(self, iou_threshold=0.0):
        ### Currently adopting maxIoU
        ### Returning indices of images and corresponding mask sets for each 3D proposals using IoU
        imgs = []
        highlight_masks = []   
        for mk3d in tqdm(self.instance3d):
            img = []
            mask_id = []
            for i in range(len(self.pcd_list)):
                frame = self.pcd_list[i]

                npixels = frame['mapping'][torch.where(mk3d==1)[0]][:,3].sum().item() # related pixels
                if npixels < 5:
                    continue
                indices =  torch.where(frame['mapping'][torch.where(mk3d==1)[0]][:,3]==1)[0]
                sieve = torch.zeros_like(frame['masks'][0])
                sieve = sieve.to(torch.bool)
                r = frame['mapping'][torch.where(mk3d==1)[0]][indices][:,[1]]
                c = frame['mapping'][torch.where(mk3d==1)[0]][indices][:,[2]]

                expanded = sieve.unsqueeze(0).expand_as(frame['masks']).cuda()
                expanded[:, r, c] = True
                # Sieve masks with 2D masks
                mask = frame['masks'].cuda()
                logicAND = mask & expanded
                logicOR = mask | expanded
                intersection = torch.logical_and(logicAND, logicOR).sum(dim=(1, 2)).float()
                union = torch.logical_or(logicAND, logicOR).sum(dim=(1, 2)).float()
                iou = intersection / union # IoU tensor of given projected points on images
                idx = torch.where(iou>=0.0)[0] # Take every 2D mask of that view
                # idx = torch.argmax(iou)
                # if iou[idx].item() == 0.0:
                #     continue

                img.append(i)
                mask_id.append(idx)
                torch.cuda.empty_cache()
            # Record for each 3D instance mask
            imgs.append(img)
            highlight_masks.append(mask_id)
        self.imgs = imgs
        self.highlight_masks = highlight_masks

    def pointwise_aggregator(self, scene_id, cfg):
        '''
            CLIP feature aggregator using weighted Superpoints
        '''
        # pcd_list = self.pcd_list
        for i in range (len(self.pcd_list)):
            self.pcd_list[i]['masks'] = []
        for (obj_id, inst) in enumerate(tqdm(self.instance3d)):
            
            target = self.storage2d[obj_id]
            msk_set = target[0]['mask2d']
            img_ids = target[0]['im_mask2d_id']
            for (iter, id) in enumerate(img_ids):
                self.pcd_list[id]['masks'].append(torch.tensor(msk_set[iter]).squeeze(0))
        for i in range (len(self.pcd_list)):
            self.pcd_list[i]['masks'] = torch.stack(self.pcd_list[i]['masks'])
        # Image Alignment
        self.align_majority()
        self.feature_bank = self.offline_images_batch(scene_id, cfg)


    def offline_images_batch(self, scene_id, cfg):
        self.offline_query_class = []
        self.offline_query_prob = []        
        print('---OFFLINE PROCESSING---')
        image_set = []
        seg_masks=[]
        img_id = []
        interval = []
        cnt = 0
        for i in range(len(self.pcd_list)):
            image_set.append(self.pcd_list[i]['image'])
            interval.append(cnt)
            for mask in self.pcd_list[i]['masks']:
                seg_masks.append(mask)
                img_id.append(i)
                cnt += 1
            interval.append(cnt)
        
        llm_feature_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.llm_feature)
        os.makedirs(llm_feature_path, exist_ok=True)
        llm_feature_path = os.path.join(llm_feature_path, scene_id + '.pth')

        embeddings = self.model.process_multiple_image(image_set, seg_masks, img_id, interval, self.imgs, self.highlight_masks)
        # temporary offload to CPU for cuda ops
        self.model.class_generator.base_model = self.model.class_generator.base_model.cpu()
        torch.cuda.empty_cache()

        feat_bank = torch.zeros((self.points.shape[0], 32, 768), dtype=torch.float16).cuda() # QFormer Channel
        counter = torch.ones((self.points.shape[0], 32, 768)).cpu()
        check = torch.zeros((self.points.shape[0], 32, 768)).cpu()
        cnt = 0
        it = 0
        print('Aggregating features')
        for i in trange(len(self.pcd_list)):
            frame = self.pcd_list[i]
            mapping = frame['mapping']
            idx = torch.where(mapping[:,3] == 1)[0]
            start_frame = interval[it]
            end_frame = interval[it + 1]
            # pred_masks = BitMasks(self.pcd_list[i]['masks'])
            pred_masks = self.pcd_list[i]['masks']
            start_tok = 0
            end_tok =  32
            while start_tok < end_tok:
                start_tokk = min(start_tok + 1, end_tok)
                final_feat = torch.einsum("qcd,qhw->cdhw", torch.stack(embeddings[start_frame:end_frame])[:, start_tok: start_tokk].to(torch.float16).cuda(), pred_masks.to(torch.float16).cuda())
                if start_tok == 0:
                    jdx = torch.where(final_feat[0, 0, mapping[idx, 1], mapping[idx, 2]]!=0)[0].cpu()
                    counter[idx[jdx]] += 1
                    check[idx[jdx]] = 1
                    del jdx
                torch.cuda.empty_cache()
                startb = 0
                endb = 768
                while startb < endb:
                    startbb = min(startb + 10, endb)
                    feat_bank[idx, start_tok : start_tokk,startb:startbb] += final_feat[:, startb : startbb, mapping[idx, 1], mapping[idx, 2]].cuda().permute(2,0,1)
                    torch.cuda.empty_cache()
                    startb = startbb
                start_tok = start_tokk
                del final_feat
            
            it += 2
            cnt += 1

        counter = counter - check # avoid divided by 0
        feat_bank = feat_bank.cpu() 
        counter = counter.cpu() 
        feat_bank/=counter
        torch.save({'feat':feat_bank}, llm_feature_path)
        torch.cuda.empty_cache()
        # GPU load for querying
        self.model.class_generator.base_model = self.model.class_generator.base_model.cuda()
        return feat_bank

    def refine_freevocab(self, scene_id, cfg, genfeature = True):
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
        self.instance3d = dic['ins']
        self.obj_id = dic['obj_bank']
        try:
            self.instance3d = torch.stack([torch.tensor(rle_decode(ins)) for ins in self.instance3d])
        except:
            pass        

        if genfeature:
            #NOTE: 2D preparation
            save_dir_2d = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output, scene_id)
            file_names = os.listdir(save_dir_2d)
            file_names.sort()
            print("-------Prepare 2D Files-------")
            self.storage2d = []
            for filename in tqdm(file_names):
                file = open(os.path.join(save_dir_2d, filename), "rb")
                self.storage2d.append(pickle.load(file))
            #NOTE: 2D preparation Detic
            # file_names = os.path.join('./data/Scannet200/Scannet200_2D_5interval/val/maskDetic', sceen_id + '.pth')
            # print("-------Prepare 2D Files-------")
            # self.storage2d = torch.load(file_names)
            self.feature_bank = self.pointwise_aggregator(scene_id, cfg)
        else:
            llm_feature_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.llm_feature)
            llm_feature_path = os.path.join(llm_feature_path, scene_id + '.pth')
            # DETIC Feature
            # llm_feature_path = os.path.join('../nips25/Queryable_FreeVocab/result_pointwise_sub/osm_feature_scannetpp/detic', scene_id + '.pth')
            self.feature_bank = torch.load(llm_feature_path)['feat']


        return self.feature_bank

    def get_final_instance(self, scene_id, cfg):

        confidences = []
        categories = []

        if len(self.class_names) < 100: # scannetpp
            sub_path = f"/root/3dllm/sub/{scene_id}.pth"
            sub_idx = torch.load(sub_path)["sub"]
            sub_inst3D = self.instance3d[:,sub_idx]
            instance3d = sub_inst3D >= 0.5        
        else: # scannet200
            data_path3D = "/root/3dllm/minhlnh/FreeVocab-3DIS/data/Scannet200/Scannet200_3D/class_ag_res_200_isbnetfull"
            scene_path3D = os.path.join(data_path3D, scene_id+'.pth')
            pred_mask3D = torch.load(scene_path3D)
            masks3D = pred_mask3D['ins']
            masks3D = torch.tensor(masks3D)
            # instance3d = torch.cat([self.instance3d, masks3D], dim = 0)       
            instance3d = self.instance3d

        ### Query class name from pc feature offline
        confidences = []
        categories = []
        print('Processing per 3D proposals')
        for i in trange(instance3d.shape[0]):
            # Per 3D proposal
            cond1 = (instance3d[i]==True)
            cond2 = (self.feature_bank[:,0,0] != 0)
            feat = self.feature_bank[cond1 & cond2].to(torch.float16)
            if feat.shape[0] == 0:
                confidences.append(0)
                categories.append('None')  
                continue
            pred_text, conf = self.model.query_llm(feat)
            confidences.append(conf)
            categories.append(pred_text)        
        return self.instance3d, confidences, categories
