from .detector3d_template import Detector3DTemplate
from pcdet.models.backbones_3d.clip_spconv_backbone import Clip_VoxelResBackBone8x
from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE
import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.spconv_utils import replace_feature, spconv
from nuscenes.nuscenes import NuScenes


CHECK_VIZ = True


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
        
        self.text_set = ['vegetation', 'road', 'street', 'sky', 'tree', 'building', 'house', 'skyscaper',
              'wall', 'fence', 'sidewalk', 'terrain', 'driveable_surface', 'manmade', 'car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'] # background_index = 0 ~ 12 , object_index = 13 ~ 22
        
        
        
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
        
        
        
        
        #################### Encoder Decoder Visualization 비교 ################################             
                 
                 
        if CHECK_VIZ:
            
            self.text_feats = self.text_extractor(self.text_set, device=self.device)             
            

            import numpy as np
            import matplotlib.pyplot as plt
            color_map = {
            'vegetation': [1, 0, 0],       # Red
            'road': [1, 0.5, 0],     # Orange
            'street': [1, 1, 0],       # Yellow
            'sky': [0, 0, 1],       # Blue
            'tree': [0, 1, 0],       # Green
            'building': [0.29, 0, 0.51], # Indigo
            'house': [0.56, 0, 1],    # Violet
            'skyscaper': [0, 1, 1],       # Cyan
            'wall': [1, 0, 1],       # Magenta
            'fence': [0.5, 0.5, 0.5],# Grey
            'sidewalk': [0.75, 0.25, 0.25], # Brown
            'terrain': [1, 0.75, 0.8], # Pink
            'driveable_surface': [0.5, 0, 0],    # Maroon
            'manmade': [1, 1, 0.5],    # Light Yellow   index 0~13 -> background
            'car': [0.5, 0.5, 0],  # Olive
            'truck': [0, 0.5, 0.5],  # Teal
            'construction_vehicle': [0.5, 0, 0.5],  # Purple
            'bus': [0.75, 0.75, 0.75], # Light Grey
            'trailer': [0.25, 0.25, 0.75], # Slate Blue
            'barrier': [0, 0.75, 0],   # Lime
            'motorcycle': [0.75, 0, 0],   # Crimson
            'bicycle': [1, 0.5, 0.5],  # Light Coral
            'pedestrian': [0.5, 1, 0.5],  # Light Green
            'traffic_cone': [0.5, 0.5, 1],  # Light Blue index 14 ~ 23 -> objects
            }
        
        
            color_map_list = list(color_map.keys())
            
            clip_spatial_features = batch_dict["clip_input_sp_tensor"].dense()
            N, C, D, H, W = clip_spatial_features.shape
            bev_image_clip_features = clip_spatial_features.view(N, C * D, H, W)              
                

            
            batch_size, clip_voxel_coords = batch_dict['batch_size'], batch_dict['clip_voxel_coords']
            recon_sp_clip_features = spconv.SparseConvTensor(
                features=recon_features,
                indices=clip_voxel_coords.int(),
                spatial_shape= np.array(self.model_cfg['CLIP_FEATURE_SHAPE']),
                batch_size=batch_size
            )
            recon_clip_spatial_features = recon_sp_clip_features.dense()
            N, C, D, H, W = recon_clip_spatial_features.shape
            recon_bev_image_clip_features = recon_clip_spatial_features.view(N, C * D, H, W)             
            
            
        
            B, C, H, W = bev_image_clip_features.shape
            image_clip_features_flatten = bev_image_clip_features.contiguous().flatten(2).squeeze().to(self.device)
            image_clip_features_norm = (image_clip_features_flatten/(image_clip_features_flatten.norm(dim=0, keepdim=True)+1e-5)).half()
            
            class_preds = (self.text_feats @ image_clip_features_norm) # text_clip_features # background_index = 0 ~ 12 , object_index = 13 ~ 22
            max_score, max_indices = torch.max(class_preds, dim=0)
            score_masked_indices = (max_score>0.02).nonzero(as_tuple=True)[0]
            
            
            height_index = (score_masked_indices // W).int()
            width_index = (score_masked_indices % W).int()
            spatial_infos = torch.stack((height_index, width_index), dim=1).cpu().detach().numpy()
            bev_vis_clip_features = np.ones((H, W, 3))
            numpy_class_idx = np.asarray(max_indices[score_masked_indices].cpu().detach().numpy(),dtype=int)
            for idx in range(len(numpy_class_idx)):
                bev_vis_clip_features[spatial_infos[idx,0],spatial_infos[idx,1],:] = color_map[color_map_list[numpy_class_idx[idx]]]
            clip_flipped_feature = torch.flip(torch.tensor(bev_vis_clip_features), [1])
            clip_rotated_feature = torch.flip(torch.transpose(clip_flipped_feature, 0, 1), [0]).cpu().detach().numpy()
            plt.clf()         
            plt.imshow(clip_rotated_feature)
            plt.savefig("/home/OpenPCDet/visualization/prediction/kitti/origin/with_bg_prediction_original.png",dpi=300, bbox_inches='tight')
            plt.axis('off')
            plt.clf() 
            plt.close('all')
        
        
        
            B, C, H, W = recon_bev_image_clip_features.shape
            recon_image_clip_features_flatten = recon_bev_image_clip_features.contiguous().flatten(2).squeeze().to(self.device)
            recon_image_clip_features_norm = (recon_image_clip_features_flatten/(recon_image_clip_features_flatten.norm(dim=0, keepdim=True)+1e-5)).half()
            
            recon_class_preds = (self.text_feats @ recon_image_clip_features_norm) # text_clip_features # background_index = 0 ~ 12 , object_index = 13 ~ 22
            recon_max_score, recon_max_indices = torch.max(recon_class_preds, dim=0)
            recon_score_masked_indices = (recon_max_score>0.02).nonzero(as_tuple=True)[0]
            
            
            recon_height_index = (recon_score_masked_indices // W).int()
            recon_width_index = (recon_score_masked_indices % W).int()
            recon_spatial_infos = torch.stack((recon_height_index, recon_width_index), dim=1).cpu().detach().numpy()
            recon_bev_vis_clip_features = np.ones((H, W, 3))
            recon_numpy_class_idx = np.asarray(recon_max_indices[recon_score_masked_indices].cpu().detach().numpy(),dtype=int)
            for idx in range(len(recon_numpy_class_idx)):
                recon_bev_vis_clip_features[recon_spatial_infos[idx,0],recon_spatial_infos[idx,1],:] = color_map[color_map_list[recon_numpy_class_idx[idx]]]
            recon_clip_flipped_feature = torch.flip(torch.tensor(recon_bev_vis_clip_features), [1])
            recon_clip_rotated_feature = torch.flip(torch.transpose(recon_clip_flipped_feature, 0, 1), [0]).cpu().detach().numpy()
            plt.clf()         
            plt.imshow(recon_clip_rotated_feature)
            plt.savefig("/home/OpenPCDet/visualization/prediction/kitti/recon/with_bg_prediction_reconstruction.png",dpi=300, bbox_inches='tight')
            plt.axis('off')
            plt.clf() 
            plt.close('all')
 
 
 
            # self.nusc = NuScenes(version='v1.0-trainval', dataroot='/home/OpenPCDet/data/nuscenes/v1.0-trainval', verbose=True)

            # token_id = batch_dict['metadata'][0]['token']
            # my_sample = self.nusc.get('sample', token_id)
            # self.nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=1, underlay_map=True, show_lidarseg=True,
            #                 show_lidarseg_legend=True, out_path='/home/OpenPCDet/visualization/groundtruth/gt.jpg')
            # plt.close('all')
            # plt.clf()
 
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[obj], markersize=8) for obj in color_map_list]
            labels = list(color_map.keys())
            fig_legend = plt.figure(figsize=(10, 2))
            plt.figlegend(handles, labels, loc='center', ncol=4, fontsize='small')
            plt.savefig("/home/OpenPCDet/visualization/prediction/with_bg_prediction_legend.png", dpi=300, bbox_inches='tight')
            plt.close(fig_legend)
 
 
        
            ########################################################################################
        
        
        
        
        
        l2loss = self.l2_loss(recon_features_norm, clip_features_norm)
        cosloss = self.cos_loss(recon_features_norm, clip_features_norm)
        
        return l2loss, cosloss

    def l2_loss(self, network_output, gt):
        return ((network_output - gt) ** 2).mean()

    def cos_loss(self, network_output, gt):
        return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()
    
    
    
    def text_extractor(self, text_set, device):
        
        model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
        model = model.to(device)
        text_set = [f"a {c} in a scene" for c in text_set]
        text_set = clip.tokenize(text_set).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_set)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features