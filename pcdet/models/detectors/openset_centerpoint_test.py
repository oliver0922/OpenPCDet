from .detector3d_template import Detector3DTemplate
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.clip_spconv_backbone import Clip_VoxelResBackBone8x
from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE
from ...utils.spconv_utils import  spconv
import numpy as np
from nuscenes.nuscenes import NuScenes



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
        

        self.text_set = ['vegetation', 'road', 'street', 'sky', 'tree', 'building', 'house', 'skyscaper',
              'wall', 'fence', 'sidewalk', 'terrain', 'driveable_surface', 'manmade', 'car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'] # background_index = 0 ~ 12 , object_index = 13 ~ 22
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_encoder=  Clip_VoxelResBackBone8x(
                                                    model_cfg=model_cfg.BACKBONE_3D,
                                                    input_channels=768,
                                                    grid_size= self.dataset.grid_size,
                                                    voxel_size=self.dataset.voxel_size,
                                                    point_cloud_range=dataset.point_cloud_range,
                                                    recon = True
                                                    ).cuda()
        
        self.nusc = NuScenes(version='v1.0-trainval', dataroot='/home/OpenPCDet/data/nuscenes/v1.0-trainval', verbose=True)

    def forward(self, batch_dict):
        
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)                 
       
         
        batch_dict = self.clip_encoder(batch_dict)
        clip_encoded_spconv_tensor = batch_dict["clip_input_sp_tensor"]
        clip_spatial_features = clip_encoded_spconv_tensor.dense()
        N, C, D, H, W = clip_spatial_features.shape
        bev_clip_spatial_features = clip_spatial_features.view(N, C * D, H, W)
        

        
        
        self.text_feats = self.text_extractor(self.text_set, device=self.device)    
            
        self.index = self.prediction_class(batch_dict, bev_clip_spatial_features, self.text_feats)
        
        
        
            
        loss, tb_dict, disp_dict = self.get_training_loss()

        ret_dict = {
                'loss': loss
            }
        return ret_dict, tb_dict, disp_dict
    
    
    
    def text_extractor(self, text_set, device):
        
        model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
        model = model.to(device)
        text_set = [f"a {c} in a scene" for c in text_set]
        text_set = clip.tokenize(text_set).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_set)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    
    def prediction_class(self, batch_dict, bev_image_clip_features, text_clip_features):
        
        
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
        
        
        
        object_color_map = {'car': [0.5, 0.5, 0],  # Olive
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
        
        
        
        
        B, C, H, W = bev_image_clip_features.shape
        image_clip_features_flatten = bev_image_clip_features.contiguous().flatten(2).squeeze().to(self.device)
        image_clip_features_norm = (image_clip_features_flatten/(image_clip_features_flatten.norm(dim=0, keepdim=True)+1e-5)).half()
        
        class_preds = (text_clip_features @ image_clip_features_norm) # text_clip_features # background_index = 0 ~ 12 , object_index = 13 ~ 22
        max_score, max_indices = torch.max(class_preds, dim=0)
        score_masked_indices = (max_score>0.02).nonzero(as_tuple=True)[0]
        
        
        bg_classes_indices = torch.tensor([i for i in range (14)])
        bg_mask = ~torch.isin(max_indices, bg_classes_indices.cuda())
        masked_score = max_score>0.05
        bg_mask = torch.logical_and(bg_mask,masked_score)
        bg_filtered_indices = bg_mask.nonzero(as_tuple=False).squeeze()
        
        
        
        ###################### visualization ########################################################################
        
        ########## clip feature bev visualization ##########
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
        plt.savefig("/home/OpenPCDet/visualization/prediction/with_bg_prediction.png",dpi=300, bbox_inches='tight')
        plt.axis('off')
        plt.clf() 
        plt.close('all')
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[obj], markersize=8) for obj in color_map_list]
        labels = list(color_map.keys())
        fig_legend = plt.figure(figsize=(10, 2))
        plt.figlegend(handles, labels, loc='center', ncol=4, fontsize='small')
        plt.savefig("/home/OpenPCDet/visualization/prediction/with_bg_prediction_legend_pedestrian.png", dpi=300, bbox_inches='tight')
        plt.close(fig_legend)


        ########################back ground removal clip feature #####################################################
        
        
        bg_rm_height_index = (bg_filtered_indices // W)
        bg_rm_width_index = (bg_filtered_indices % W)
        bg_rm_spatial_infos = torch.stack((bg_rm_height_index, bg_rm_width_index), dim=1).cpu().detach().numpy()
        bev_bg_rm_vis_clip_features = np.ones((H, W, 3))
        bg_rm_numpy_class_idx = np.asarray(max_indices[bg_filtered_indices].cpu().detach().numpy(),dtype=int)
        
        for idx in range(len(bg_rm_numpy_class_idx)):
            bev_bg_rm_vis_clip_features[bg_rm_spatial_infos[idx,0], bg_rm_spatial_infos[idx,1], :] = color_map[color_map_list[bg_rm_numpy_class_idx[idx]]]
        
        clip_flipped_feature_bg_rm = torch.flip(torch.tensor(bev_bg_rm_vis_clip_features), [1])
        clip_rotated_feature_bg_rm = torch.flip(torch.transpose(clip_flipped_feature_bg_rm, 0, 1), [0]).cpu().detach().numpy()
        plt.imshow(clip_rotated_feature_bg_rm)
        plt.savefig("/home/OpenPCDet/visualization/prediction/without_bg_prediction.png",dpi=300, bbox_inches='tight')
        plt.axis('off')
        plt.clf() 
        plt.close('all')
        
        object_color_map_list = color_map_list[14:]
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[obj], markersize=8) for obj in object_color_map_list]
        labels = list(object_color_map.keys())
        fig_legend = plt.figure(figsize=(10, 2))
        plt.figlegend(handles, labels, loc='center', ncol=4, fontsize='small')
        plt.savefig("/home/OpenPCDet/visualization/prediction/without_bg_prediction_legend.png", dpi=300, bbox_inches='tight')
        plt.clf() 
        plt.close(fig_legend)

        
        
        
        ############# gt bev visualization #################
        
        CLASS_NAMES_EACH_HEAD= [
            ['car'], 
            ['truck', 'construction_vehicle'],
            ['bus', 'trailer'],
            ['barrier'],
            ['motorcycle', 'bicycle'],
            ['pedestrian', 'traffic_cone'],
        ]       
        
        bev_vis_gt_features = np.ones((H, W, 3))
        for class_head_idx in range(6):
            gt_object_index = batch_dict['gt_inds'][class_head_idx]
            mask = gt_object_index>0
            masked_gt_object_index = gt_object_index[mask]
            num_objects = len(masked_gt_object_index)
            gt_height_index = (masked_gt_object_index // W)
            gt_width_index = (masked_gt_object_index % W)
            gt_spatial_infos = torch.stack((gt_height_index, gt_width_index), dim=1).cpu().detach().numpy()
            
            sub_class_list = list()
            for obj_idx in range(num_objects):
                sub_class = batch_dict['gt_class_inds'][class_head_idx]
                sub_class_idx = sub_class[0][obj_idx][-1]-1
                sub_class_list.append(CLASS_NAMES_EACH_HEAD[class_head_idx][sub_class_idx.int()])
            
            for obj_idx in range(num_objects):
                bev_vis_gt_features[gt_spatial_infos[obj_idx,0],gt_spatial_infos[obj_idx,1], :] = color_map[sub_class_list[obj_idx]]
        gt_flipped_feature = torch.flip(torch.tensor(bev_vis_gt_features), [1])
        gt_rotated_feature = torch.flip(torch.transpose(gt_flipped_feature, 0, 1), [0]).cpu().detach().numpy()
        plt.imshow(gt_rotated_feature)
        #####################################################
        
        ################### render sample ######################
        token_id = batch_dict['metadata'][0]['token']
        my_sample = self.nusc.get('sample', token_id)
        self.nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=1, underlay_map=True, show_lidarseg=True,
                        show_lidarseg_legend=True, out_path='/home/OpenPCDet/visualization/groundtruth/gt.jpg')
        plt.close('all')
        plt.clf()
        ###################################################
        
        #######################################################
        
        return class_preds
    
    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
