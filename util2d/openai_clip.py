import torch
import numpy as np
import clip

class CLIP_OpenAI:
    def __init__(self, cfg):
        # CLIP
        # breakpoint()
        # 
        # self.clip_adapter, self.clip_preprocess = clip.load(cfg.segmenter2d.clip_model, device = 'cuda')
        self.clip_adapter, self.clip_preprocess = clip.load("/root/3dllm/ViT-L-14-336px.pt", device = 'cuda')
        
        self.dim = 768
        print('------- Loaded CLIP ViT-L/14 336px OpenAI -------')