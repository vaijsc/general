import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from scannetv2_inst_eval import ScanNetEval
# from scannetv2_inst_eval_freevocab import ScanNetEval

# from scannetv2_freevocab_inst_eval import ScanNetFreeVocabEval

import os
from loader3d.scannet200 import INSTANCE_CAT_SCANNET_200
from loader3d.scannetpp import INSTANCE_BENCHMARK84_SCANNET_PP, SEMANTIC_INSTANCE_BENCHMARK84_SCANNET_PP
from tqdm import tqdm, trange


# Hungarian Match
from scipy.optimize import linear_sum_assignment

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

## Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
## Load ScannetEval


# freevocab_eval = ScanNetFreeVocabEval(class_labels = INSTANCE_CAT_SCANNET_200)


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

def class_similarity(nouns_listA, nouns_listB):
    sentenceA = [] # predicts
    sentenceB = [] # gt label
    for name in nouns_listA:
        sentenceA.append('This is a ' + name)
    for name in nouns_listB:
        sentenceB.append('This is a ' + name)

    # Tokenize sentences
    encoded_inputA = tokenizer(sentenceA, padding=True, truncation=True, return_tensors='pt')
    encoded_inputB = tokenizer(sentenceB, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_outputA = model(**encoded_inputA)
        model_outputB = model(**encoded_inputB)

    sentence_embeddingsA = mean_pooling(model_outputA, encoded_inputA['attention_mask'])
    sentence_embeddingsA = F.normalize(sentence_embeddingsA, p=2, dim=1)
    sentence_embeddingsB = mean_pooling(model_outputB, encoded_inputB['attention_mask'])
    sentence_embeddingsB = F.normalize(sentence_embeddingsB, p=2, dim=1)

    A = sentence_embeddingsA.cuda()
    B = sentence_embeddingsB.cuda()
    score = torch.mm(A, B.permute(1, 0))
    return score

if __name__=='__main__':


    data_path = "../freevocab_exp_scannetpp/version_dp_maximum_score_0.6_n_spp_div4/freevocab_results"
    dataset = "scannetpp"
    # data_path = "/root/3dllm/vis/results_pointwise_osm_isbnet_detic_scannetpp"    

    if dataset == "scannet200":
        scan_eval = ScanNetEval(class_labels = INSTANCE_CAT_SCANNET_200)
        pcl_path = './data/Scannet200/Scannet200_3D/val/groundtruth'
    elif dataset == "scannetpp":
        scan_eval = ScanNetEval(class_labels = INSTANCE_BENCHMARK84_SCANNET_PP, dataset_name="scannetpp_benchmark_instance")
        pcl_path = '../../groundtruth_benchmark_instance'
        sub_path = "../../sub"


    scenes = sorted([s for s in os.listdir(data_path) if s.endswith('.pth')])

    # for scene in scenes1:
    #     print(f"Processing {scene}")
    
    gtsem = []
    gtinst = []
    res = [] #ScannetV2

    # scenes = ["scene0423_00.pth"] #
    for scene in tqdm(scenes):

        if dataset == "scannet200":
            gt_path = os.path.join(pcl_path, scene)
            loader = torch.load(gt_path)
            sem_gt, inst_gt = loader[2], loader[3]
        elif dataset == "scannetpp":
            gt_path = os.path.join(pcl_path, scene.replace('.pth','') + '.pth')
            loader = torch.load(gt_path)
            sub_idx = torch.load(os.path.join(sub_path, scene.replace('.pth','') + '.pth'))["sub"]
            sem_gt, inst_gt = loader[2][sub_idx], loader[3][sub_idx]
    
        gtsem.append(np.array(sem_gt).astype(np.int32))
        gtinst.append(np.array(inst_gt).astype(np.int32))
        scene_path = os.path.join(data_path, scene)
        pred_mask = torch.load(scene_path)
        try:
            masks, category, score2d, score3d = pred_mask['masks'], pred_mask['class'], pred_mask['confidence2d'], pred_mask['confidence3d']
        except:
            masks, category, score3d = pred_mask['ins'], pred_mask['name'], pred_mask['conf']
        
        # Assuming masks and category are lists of the same length
        # masks = [mask for mask, cat in zip(masks, category) if cat not in ["floor", "wall"]]
        # category = [cat for cat in category if cat not in ["floor", "wall"]]
        if dataset == "scannet200":
            scores = class_similarity(category, INSTANCE_CAT_SCANNET_200)
        elif dataset == "scannetpp":
            scores = class_similarity(category, INSTANCE_BENCHMARK84_SCANNET_PP)

        category = torch.argmax(scores, dim = 1)
        sentence_score = torch.max(scores, dim = 1)[0]
        n_mask = len(category)

        tmp = []
        for ind in range(len(masks)):
            if isinstance(masks[ind], dict):
                mask = rle_decode(masks[ind])
            else:
                mask = (masks[ind] == 1).numpy().astype(np.uint8)
            if dataset == "scannetpp":
                mask = mask[sub_idx]

            conf = 1.0

            scene_id = scene.replace('.pth', '')
            # if dataset == "scannet200":
            #     tmp.append({'scan_id': scene_id, 'label_id':0 + 1, 'conf':conf, 'pred_mask':mask, 'vocab_score': scores[ind]})
            # elif dataset == "scannetpp":
            #     tmp.append({'scan_id': scene_id, 'label_id':0 + 1, 'conf':conf, 'pred_mask':mask[sub_idx], 'vocab_score': scores[ind]})

            if sentence_score[ind].item() < 0.5:
                continue

            final_class = float(category[ind].item())
            scene_id = scene.replace('.pth', '')
            tmp.append({'scan_id': scene_id, 'label_id':final_class + 1, 'conf':conf, 'pred_mask':mask, 'vocab_score': scores[ind]})
            
        res.append(tmp)

    ###---------------------------------- OFFLINE process, for ablation----------------------------------
    # torch.save(res,'res.pth')
    # torch.save(gtsem,'gtsem.pth') 
    # torch.save(gtinst,'gtinst.pth')

    # res = torch.load('res.pth')
    # gtsem = torch.load('gtsem.pth')
    # gtinst = torch.load('gtinst.pth')
    ###----------------------------------###----------------------------------###----------------------------------

    ## Hungarian process
    scene_num = 0
    correct = 0
    total = 0
    freevocab_metric = []
    for mask_list in tqdm(res):
        cost_matrix = [] # perscene
        gt_masks = gtinst[scene_num].copy()
        gt_labels = gtsem[scene_num].copy() # shiftlabel
        if dataset == 'scannet200':
            gt_labels -= 2
        
        unique_instance_ids = np.unique(gt_masks)
        binary_matrix = (gt_masks[:, None] == unique_instance_ids).astype(bool)
        gt_label = []
        temporary_mat = np.transpose(binary_matrix)
        for mk in temporary_mat:
            gt_label.append(gt_labels[mk==True][0])
        gt_label = np.array(gt_label)
        for cmp in mask_list:
            scene_id = cmp['scan_id']
            mask = cmp['pred_mask'].astype(bool)
            score = cmp['vocab_score']
            conf = cmp['conf']

            ## IoU calc
            OR = np.logical_or(mask[:, np.newaxis], binary_matrix) 
            AND = np.logical_and(mask[:, np.newaxis], binary_matrix) 
            IoU = np.sum(AND, axis=(0))/np.sum(OR, axis=(0))
            # Score calc
            neg_id = np.where(gt_label < 0)[0]
            gt_label[neg_id] = 0
            cost = torch.sqrt(torch.tensor(IoU).cuda()*score[gt_label])
            cost[neg_id] = 0.0
            gt_label[neg_id] = 0
            cost_matrix.append(cost.cpu().numpy())

        cost_matrix = np.array(cost_matrix)
        row, col = linear_sum_assignment(-cost_matrix)
        res[scene_num] = [res[scene_num][id] for id in row] 
        # FreeVocab Metric
        freevocab_metric.append(np.sum(cost_matrix[row, col])/len(row))
        # Accuracy calc
        list1 = [int(res[scene_num][id]['label_id']) - 1 for id in range(len(row))] # prediction
        list2 = [gt_label[col[id]] for id in range(len(row))] # gt
        matching_elements = sum(1 for a, b in zip(list1, list2) if a == b)
        total_elements = len(list1)
        correct += matching_elements
        total += total_elements
        scene_num += 1
    freevocab_metric = np.average(np.array(freevocab_metric))
    print('FreeVocab metric:', freevocab_metric) # cang cao cang tot
    print('Correct Samples:', correct)
    print('Total Samples:', total)
    scan_eval.evaluate(res, gtsem, gtinst, scores)

