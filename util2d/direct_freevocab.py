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
from OmniScientModel.utils import prepare_osm, prepare_instruction
import torchvision.transforms as T


INPUT_SIZE = 1120
CONTEXT_ENLARGE_RATIO = 0.5
# The float16 affects final results slightly. Results in paper are obtained with float32.
TEST_DTYPE = torch.float16

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

    def save_imgs(self, imgs, masks, folder="test_imgs/"):
        
        for i in range(imgs.shape[0]):
            plt.figure(figsize=(10, 10))
            image = imgs[i]
            image = (image * 255).clamp(0, 255).byte()
            image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')
            plt.imshow(image)
            plt.axis('off')
            mask = masks[i]
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            plt.savefig(os.path.join(folder+str(i)+'.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    @torch.no_grad()
    def process_multiple_image(self, images, seg_masks, img_id, interval, imgs, highlight_masks):
        ## retrieve qformer embed ->  LLM avg from OSM from multiple (image) and list of segmentation masks  

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
        
        # Get indices for each 3D proposals
        proposal_indices = []
        for i in trange(len(imgs)):
            # Per 3D proposal
            pr_ids = []
            if len(imgs[i]) == 0:
                proposal_indices.append(pr_ids)
                continue
            for j in range(len(imgs[i])): # each corresponding frame
                for k in highlight_masks[i][j]: # each corresponding mask of the frame
                    frame_id = imgs[i][j]
                    pr_ids.append(img_id.index(frame_id) + k)
            proposal_indices.append(pr_ids)
        
        '''
        MULTI version: having batch size
        batch of image and mask  (1 image, n mask)
        duplicate into n(1 image, 1 mask) per forwarding

        img_id = [nMask], keeping track mask belong to which image
        '''
        qformer_embed = torch.zeros((len(segmentation_masks_batch), 32, 768), dtype= torch.float16) # qformer channel
        print('Querying OSM...')
        img_ids = torch.tensor(img_id)
        images_batchs = torch.stack(images_batch)
        qformer_input_ids_batchs = torch.cat(qformer_input_ids_batch)
        qformer_attention_mask_batchs = torch.cat(qformer_attention_mask_batch)
        input_ids_batchs = torch.cat(input_ids_batch)
        attention_mask_batchs = torch.cat(attention_mask_batch)
        segmentation_masks_batchs = torch.cat(segmentation_masks_batch)
        context_mask_batch = torch.cat(context_mask_batch)

        for i in trange(len(imgs)): # loop through each proposal 3D
            ### Generating class
            with torch.no_grad():
                batch_indices = proposal_indices[i]
                if len(batch_indices) == 0:
                    continue
                    
                generated_output_qformer = self.class_generator.generate_1n_avg_qformer(
                    img_idd =  img_ids[batch_indices].tolist(),
                    pixel_values=images_batchs[batch_indices].cuda().to(torch.float16),
                    qformer_input_ids=qformer_input_ids_batchs[batch_indices].cuda(),
                    qformer_attention_mask=qformer_attention_mask_batchs[batch_indices].cuda(),
                    input_ids=input_ids_batchs[batch_indices].cuda(),
                    attention_mask=attention_mask_batchs[batch_indices].cuda(),
                    cache_image_embeds=True,
                    segmentation_mask=segmentation_masks_batchs[batch_indices].cuda(),
                    input_context_mask=context_mask_batch[batch_indices].cuda(),
                    dataset_type="any",
                    max_new_tokens=16,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    is_average_crossattention=False,
                    ).cpu() # cpu for sake of mem
                qformer_embed[batch_indices] = generated_output_qformer
            torch.cuda.empty_cache()
        #### DONE HERE
        embeddings = []
        for i in range(0, len(interval) - 1, 2):
            embeddings.append(qformer_embed[interval[i]:interval[i + 1]])
            
        folder = os.path.join(save_dir, "ins_feat")
        os.makedirs(folder, exist_ok=True)
        dir_osm_feat = os.path.join(folder, f"{scene_id}.pth")
        torch.save({"feat": embeddings}, dir_osm_feat)

        
        final_class = []
        conf3d = []
        for i in trange(len(imgs)):
            # Per 3D proposal
            if len(imgs[i]) == 0:
                final_class.append(None)
                continue
            representatives = [] # representation qformer vectors for the 3D proposal
            for j in range(len(imgs[i])):
                for k in highlight_masks[i][j]:
                    representatives.append(embeddings[imgs[i][j]][k].unsqueeze(0))
            generated_output = self.class_generator.generate_1n_avg_language(
                                    qformer_output=torch.cat(representatives).cuda(),
                                    input_ids=torch.cat(input_ids_batch[0:1]).cuda(),
                                    attention_mask=torch.cat(attention_mask_batch[0:1]).cuda(),
                                    dataset_type="any",
                                    max_new_tokens=16,
                                    num_beams=1,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    is_average_crossattention=False,
                                    )
            generated_text = generated_output["sequences"][0]
            gentext = self.processor.tokenizer.decode(generated_text).split('</s>')[1].strip()
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
            final_class.append(gentext)
            conf3d.append(pred_prob)
        return final_class, conf3d
   
    def get_context_mask(self, mask, input_size, enlarge_ratio=0.5):
        if mask.sum() == 0:
            raise ValueError("Got an empty mask!")
        
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

    def get_pred_class_and_prob(self, image, binary_mask, cache_flag=True):
        image = Image.fromarray(image).convert("RGB")
        if min(image.size) == max(image.size):
            image = T.functional.resize(image, size=self.input_size, interpolation=T.functional.InterpolationMode.BICUBIC)
        else: # in our case, always this case
            image = T.functional.resize(image, size=self.input_size - 1, max_size=self.input_size, interpolation=T.functional.InterpolationMode.BICUBIC)
        
        image_for_seg = np.array(image)
        # pad to input_size x input_size
        padded_image = np.zeros(shape=(self.input_size, self.input_size, 3), dtype=np.uint8)
        padded_image[:image_for_seg.shape[0], :image_for_seg.shape[1]] = image_for_seg
        image = Image.fromarray(padded_image)
        image = self.processor(images=[image], return_tensors="pt")["pixel_values"].view(1, 3, self.input_size, self.input_size)

        binary_mask = torch.nn.functional.interpolate(torch.from_numpy(binary_mask).to(torch.float16).unsqueeze(1).cuda(),size=(self.input_size, self.input_size), mode='bicubic')
        binary_mask = (binary_mask > 0.5) # booling the mask

        context_mask = self.get_context_mask(binary_mask, INPUT_SIZE, CONTEXT_ENLARGE_RATIO).view(1, 1, INPUT_SIZE, INPUT_SIZE)

        qformer_feat = self.class_generator.generate_qformer_output(
            pixel_values=image.cuda().to(TEST_DTYPE),
            qformer_input_ids=self.qformer_lang_x["input_ids"].cuda(),
            qformer_attention_mask=self.qformer_lang_x["attention_mask"].cuda(),
            input_ids=self.lang_x["input_ids"].cuda(),
            attention_mask=self.lang_x["attention_mask"].cuda(),
            cache_image_embeds=cache_flag,
            segmentation_mask=binary_mask.cuda(),
            input_context_mask=context_mask.cuda(),
            # 12 could be too much considering most gt are single word
            max_new_tokens=12,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True
        )

        return qformer_feat

class FreeVocab_Direct:
    def __init__(self, cfg, class_names):
        # OpenAI CLIP
        self.device = "cuda:0"
        self.freevocab_model = OSM()
        self.cfg = cfg
        self.class_names = class_names
        # with torch.no_grad(), torch.cuda.amp.autocast():
        #     text_features = self.clip_model.clip_adapter.encode_text(clip.tokenize(class_names).to(self.device))
        #     text_features /= text_features.norm(dim=-1, keepdim=True)
        # self.text_features = text_features
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
        
        final_feature= []
        for rr in trange(instance3d.shape[0]):
            msk_set = mask_bank[rr]
            images = [self.pcd_list[id]['image'] for id in id_bank[rr]]
            mappings = [self.pcd_list[id]['mapping'] for id in id_bank[rr]]

            H, W = images[0].shape[0], images[0].shape[1]
            # Open Vocab
            feat = []
            for i in range(len(images)): # 5candidate
                img = np.array(images[i])
                msk = torch.tensor(msk_set[i])
                qformer_feat = self.freevocab_model.get_pred_class_and_prob(images[i], msk)
                feat.append(qformer_feat)
            feat = torch.stack(feat, 0)
            feat = torch.mean(feat, keepdim = True)
            final_feature.append(feat)
        
        confidences = []
        categories = []
        sub_path = f"/root/3dllm/sub/{self.scene_id}.pth"
        sub_idx = torch.load(sub_path)["sub"]

        sub_inst3D = self.instance3d[:,sub_idx]

        instance3d_spp = sub_inst3D >= 0.5
        for i in range(instance3d_spp.shape[0]):
            generated_output = self.freevocab_model.class_generator.decode_qformer(
                qformer_output=final_feature[i].to(TEST_DTYPE),
                input_ids=self.freevocab_model.lang_x["input_ids"].cuda(),
                attention_mask=self.freevocab_model.lang_x["attention_mask"].cuda(),
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated_text = generated_output["sequences"][0]
            scores = generated_output["scores"]
            pred_class = self.freevocab_model.processor.tokenizer.decode(generated_text).split('</s>')[1].strip()
            pred_class_tokenidx = self.freevocab_model.processor.tokenizer.encode(pred_class)
            scores = scores[:len(pred_class_tokenidx) -1] # minus one for bos token
            # matching the pred_class_tokenidx for prob computation
            temp = 1.0
            probs = [torch.nn.functional.softmax(score / temp, dim=-1) for score in scores]
            pred_prob = 1.0
            for p_idx, prob in enumerate(probs):
                pred_idx = pred_class_tokenidx[p_idx + 1]
                pred_prob *= prob[0, pred_idx].cpu().item()
            # print(f"pred_class: {pred_class}, pred_prob: {pred_prob}")
            categories.append(pred_class)
            confidences.append(pred_prob)
            # return pred_class, pred_prob

            # feat = feat[feat[:,0]!=0]
            # feat = torch.mean(feat, dim = 0)

            # final_id = (feat.to(torch.float32) @ self.text_features.T.to(torch.float32)).softmax(dim=-1)
            # final_id = torch.argmax(final_id, dim = -1).item()
            # class_ids.append(final_id)
            # categories.append(self.class_names[final_id])

        return self.instance3d, confidences, categories
