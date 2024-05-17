from .detector3d_template import Detector3DTemplate
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.clip_spconv_backbone import Clip_VoxelResBackBone8x
from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE
from ...utils.spconv_utils import  spconv
import numpy as np



class Openset_CenterPoint_test(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.dataset = dataset
        self.logger = dataset.logger
        if 'COS_LOSS_WEIGHT' in self.model_cfg.keys():
            self.cos_loss_weight = self.model_cfg.COS_LOSS_WEIGHT
        else:
            self.cos_loss_weight = 1
        
        self.predefined_label = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.text_set = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_encoder=  Clip_VoxelResBackBone8x(
                                                    model_cfg=model_cfg.BACKBONE_3D,
                                                    input_channels=768,
                                                    grid_size= self.dataset.grid_size,
                                                    voxel_size=self.dataset.voxel_size,
                                                    point_cloud_range=dataset.point_cloud_range,
                                                    recon = True
                                                    ).cuda()

    def forward(self, batch_dict):
        
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)                 
       
         
        batch_dict = self.clip_encoder(batch_dict)
        clip_encoded_spconv_tensor = batch_dict["clip_input_sp_tensor"]
        clip_spatial_features = clip_encoded_spconv_tensor.dense()
        N, C, D, H, W = clip_spatial_features.shape
        clip_spatial_features = clip_spatial_features.view(N, C * D, H, W)
        

        
        
        self.text_feats = self.text_extractor(self.text_set, device=self.device)    
            
        self.index = self.prediction_class(clip_spatial_features, self.text_feats)
        
        
        
            
        return pred_dicts, recall_dicts
    
    
    
    def text_extractor(self, text_set, device):
        
        model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
        model = model.to(device)
        text_set = [f"a {c} in a scene" for c in text_set]
        text_set = clip.tokenize(text_set).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_set)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    
    def prediction_class(self, image_clip_features, text_clip_features):
        
        
        image_clip_features_flatten = image_clip_features.contiguous().flatten(2)
        
        class_preds = (100.0 * text_clip_features @ image_clip_features_flatten.half().to(self.device).squeeze()).softmax(dim=-1)
        max_score, max_indices = torch.max(class_preds, dim=-1)
        
        
        
        
        return class_preds