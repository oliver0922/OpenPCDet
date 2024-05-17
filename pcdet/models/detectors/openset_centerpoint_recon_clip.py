from .detector3d_template import Detector3DTemplate
from pcdet.models.backbones_3d.clip_spconv_backbone import Clip_VoxelResBackBone8x
from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE

import torch
import torch.nn as nn
import torch.nn.functional as F




class Openset_CenterPoint_Recon_clip(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        if 'COS_LOSS_WEIGHT' in self.model_cfg.keys():
            self.cos_loss_weight = self.model_cfg.COS_LOSS_WEIGHT
        else:
            self.cos_loss_weight = 1
        self.module_list = self.build_networks()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_encoder=  Clip_VoxelResBackBone8x(
                                                    model_cfg=model_cfg,
                                                    input_channels=768,
                                                    grid_size= self.dataset.grid_size,
                                                    voxel_size=self.dataset.voxel_size,
                                                    point_cloud_range=dataset.point_cloud_range,
                                                    recon = True
                                                    ).cuda()
        self.encoder = nn.Sequential(
        nn.Linear(768,256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32,16),
        nn.BatchNorm1d(16),
        nn.ReLU(),                          
        )
        
        self.decoder = nn.Sequential(
        nn.Linear(16,32),
        nn.ReLU(),
        nn.Linear(32,64),
        nn.ReLU(),
        nn.Linear(64,128),
        nn.ReLU(),        
        nn.Linear(128,256),
        nn.ReLU(),
        nn.Linear(256,768),             
        )
        
        
        
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            if isinstance(cur_module,MeanVFE):
                batch_dict = self.clip_encoder(batch_dict)
                
        if self.training:
            
            tb_dict = {}
            disp_dict = {}
            l2loss, cosloss = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': l2loss + cosloss*self.cos_loss_weight
            }
            tb_dict.update({'l2loss':l2loss.item(), 'cosloss':cosloss.item()*self.cos_loss_weight })
            return ret_dict, tb_dict, disp_dict

    
    
        
    def get_training_loss(self, batch_dict):
        
    
 
        clip_features = batch_dict["clip_input_sp_tensor"].features
        encoded_clip_features = self.encoder(clip_features)
        recon_features = self.decoder(encoded_clip_features)
 
        
        clip_features_norm = (clip_features/(clip_features.norm(dim=-1, keepdim=True)+1e-5))
        recon_features_norm = (recon_features/(recon_features.norm(dim=-1, keepdim=True)+1e-5))
        
        l2loss = self.l2_loss(recon_features_norm, clip_features_norm)
        cosloss = self.cos_loss(recon_features_norm, clip_features_norm)
        
        return l2loss, cosloss

    def l2_loss(self, network_output, gt):
        return ((network_output - gt) ** 2).mean()

    def cos_loss(self, network_output, gt):
        return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()
    
    
    