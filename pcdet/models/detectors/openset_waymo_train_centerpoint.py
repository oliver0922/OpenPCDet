from .detector3d_template import Detector3DTemplate
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.clip_spconv_backbone import Clip_VoxelResBackBone8x
from pcdet.models.dense_heads.openset_center_head import Openset_CenterHead
from ...utils.spconv_utils import  spconv
import numpy as np


CHECK_VIZ = True
class Openset_Waymo_CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.dataset = dataset
        self.logger = dataset.logger
        # self.loss_type = model_cfg.LOSS
        
        if 'COS_LOSS_WEIGHT' in self.model_cfg.keys():
            self.cos_loss_weight = self.model_cfg.COS_LOSS_WEIGHT
        else:
            self.cos_loss_weight = 1
            
        if 'MASK' in self.model_cfg.keys():
            self.mask = self.model_cfg.MASK
        else:
            self.mask = False        

        self.text_set = ['vegetation', 'road', 'street', 'sky', 'tree', 'building', 'house', 'skyscaper',
              'wall', 'fence', 'sidewalk', 'terrain', 'driveable_surface', 'manmade', 'car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.excavator = ['suv','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone', 'road', 'street', 'sky', 'tree', 'building', 'house', 'skyscaper',
              'wall', 'fence', 'sidewalk', 'terrain', 'driveable_surface', 'manmade']
        
        # self.text_set = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
        #       'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_encoder=  Clip_VoxelResBackBone8x(
                                                    model_cfg=model_cfg.BACKBONE_3D,
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
            if isinstance(cur_module, Openset_CenterHead) and self.mode=='TEST':
                self.text_feats = self.text_extractor(self.excavator, device=self.device)
                batch_dict = cur_module(batch_dict, self.decoder, self.text_feats)
            else:
                batch_dict = cur_module(batch_dict)


        
        ####### only clip_feature_training ######################
        if 'clip_train' in batch_dict.keys() and self.mode=='TRAIN':
            
            
            
            ################ reconstruction pretrained model load ###############
            # checkpoint = torch.load(self.model_cfg.CLIP_PRETRAINED_PATH)
            # model_state_disk = checkpoint['model_state']
            # state_dict = self.state_dict()
            # update_model_state = {}
            
            # for key,val in model_state_disk.items():
            #     if key in state_dict and state_dict[key].shape == val.shape:
            #         update_model_state[key] = val
            
            # # state_dict.update(update_model_state)
            # self.load_state_dict(state_dict)
            # for key in state_dict:
            #     if key not in update_model_state:
            #         self.logger.info('encoder part: Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
            # self.logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))
            ######################################################################
            
            
            
            batch_dict = self.clip_encoder(batch_dict)
            
        ###################### freeze except centerpoint clip head #########################
            for cur_module in self.module_list: #centerpoint
                for param in cur_module.parameters():
                    param.requires_grad = False
            for cur_module in self.encoder: #encoder
                for param in cur_module.parameters():
                    param.requires_grad = False
            for cur_module in self.decoder: #decoder
                for param in cur_module.parameters():
                    param.requires_grad = False           
            for clip_param in self.dense_head.heads_list[6].clip.parameters():
                clip_param.requires_grad = True
                
            # freezed_modules = [] ####### 학습 가능한 layer 찾기
            # for name, param in self.named_parameters():
            #     if param.requires_grad:
            #         freezed_modules.append(name)    
            
                
        ###################################################################################                       
       
       
       
       ########## decoder 활용해서 visualization ######################
       
        if CHECK_VIZ:
            batch_dict = self.clip_encoder(batch_dict)       
            

            import numpy as np
            import matplotlib.pyplot as plt
            self.text_set = ['vegetation', 'road', 'street', 'sky', 'building',
              'wall', 'fence', 'sidewalk', 'terrain',  'car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
            color_map = {
            'vegetation': [1, 0, 0],       # Red
            'road': [1, 0.5, 0],     # Orange
            'street': [1, 1, 0],       # Yellow
            'sky': [0, 0, 1],       # Blue
            'building': [0.29, 0, 0.51], # Indigo
            'wall': [1, 0, 1],       # Magenta
            'fence': [0.5, 0.5, 0.5],# Grey
            'sidewalk': [0.75, 0.25, 0.25], # Brown
            'terrain': [1, 0.75, 0.8], # Pink
            'car': [0.5, 0.5, 0],  # Olive
            'truck': [0, 0.5, 0.5],  # Teal
            'construction_vehicle': [0.5, 0, 0.5],  # Purple
            # 'bus': [0.75, 0.75, 0.75], # Light Grey
            'bus': [0.5, 1, 0.5], # Light green
            'trailer': [0.25, 0.25, 0.75], # Slate Blue
            'barrier': [0, 0.75, 0],   # Lime
            'motorcycle': [0.75, 0, 0],   # Crimson
            'bicycle': [1, 0.5, 0.5],  # Light Coral
            # 'pedestrian': [0.5, 1, 0.5],  # Light Green
            'pedestrian': [0.75, 0.75, 0.75],  # grey
            'traffic_cone': [0.5, 0.5, 1],  # Light Blue index 14 ~ 23 -> objects
            }
            self.text_feats = self.text_extractor(self.text_set, device=self.device)             

            # color_map = {
            # 'car': [0.5, 0.5, 0],  # Olive
            # 'truck': [0, 0.5, 0.5],  # Teal
            # 'construction_vehicle': [0.5, 0, 0.5],  # Purple
            # 'bus': [0.75, 0.75, 0.75], # Light Grey
            # # 'bus': [0.5, 1, 0.5], # Light Grey
            # 'trailer': [0.25, 0.25, 0.75], # Slate Blue
            # 'barrier': [0, 0.75, 0],   # Lime
            # 'motorcycle': [0.75, 0, 0],   # Crimson
            # 'bicycle': [1, 0.5, 0.5],  # Light Coral
            # 'pedestrian': [0.5, 1, 0.5],  # Light Green
            # # 'pedestrian': [0.75, 0.75, 0.75],  # Light Green
            # 'traffic_cone': [0.5, 0.5, 1],  # Light Blue index 14 ~ 23 -> objects
            # }
            
            
        
            color_map_list = list(color_map.keys())            
            
            ######### legend ##########
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[obj], markersize=8) for obj in color_map_list]
            labels = list(color_map.keys())
            fig_legend = plt.figure(figsize=(10, 2))
            plt.figlegend(handles, labels, loc='center', ncol=4, fontsize='small')
            plt.savefig("/home/OpenPCDet/visualization/prediction/with_bg_prediction_legend_waymo.png", dpi=300, bbox_inches='tight')
            plt.close(fig_legend)
            ##########################
            
            
            
            
            
            # clip_features = batch_dict["clip_input_sp_tensor"].features
            # encoded_clip_features = self.encoder(clip_features)
            # recon_features = self.decoder(encoded_clip_features)
            
            
            
            
            
            
            clip_spatial_features = batch_dict["clip_input_sp_tensor"].dense()
            N, C, D, H, W = clip_spatial_features.shape
            bev_image_clip_features = clip_spatial_features.view(N, C * D, H, W)             
            
            
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
            plt.savefig("/home/OpenPCDet/visualization/prediction/waymo/origin/with_bg_prediction_original.png",dpi=300, bbox_inches='tight')
            plt.axis('off')
            plt.clf() 
            plt.close('all')           
            
            ############ gt #############
            points = batch_dict['points'][:, 1:]
            gt_boxes = batch_dict['gt_boxes'][0][:,:7]
            gt_labels =  batch_dict['gt_boxes'][0][:,7]
            torch.save(points, '/home/OpenPCDet/data_for_vis/kitti_train_waymo_test/points.pt')
            torch.save(gt_boxes,'/home/OpenPCDet/data_for_vis/kitti_train_waymo_test/gt_boxes.pt')
            torch.save(gt_labels,'/home/OpenPCDet/data_for_vis/kitti_train_waymo_test/gt_labels.pt')
            ####################################
            
            
            
            
            
            ########################################### reconstruction visualization ########################################
            
            # clip_features = batch_dict["clip_input_sp_tensor"].features
            # encoded_clip_features = self.encoder(clip_features)
            
            
            
            # ######################visualization for encoded bev clip features###########################
            # # encoded_sp_clip_features = spconv.SparseConvTensor(
            # #     features=encoded_clip_features,
            # #     indices= batch_dict['clip_voxel_coords'].int(),
            # #     spatial_shape= np.array(self.model_cfg['CLIP_FEATURE_SHAPE']),
            # #     batch_size=batch_dict['batch_size']
            # # )
            # # encoded_sp_clip_features_densed = encoded_sp_clip_features.dense()
            # # N, C, D, H, W = encoded_sp_clip_features_densed.shape
            # # encoded_bev_image_clip_features = encoded_sp_clip_features_densed.view(N, C * D, H, W) 
            
            # # mean_encoded_bev_image_clip_features = encoded_bev_image_clip_features[0].mean(dim=0)
            # # mean_predicted_bev_clip_feature = predicted_bev_clip_feature[0].mean(dim=0)
            # # a = predicted_bev_clip_feature[0,:,0,0]
            # # b = encoded_bev_image_clip_features[0,:,0,0]
            # #############################################################################################
            
            
            
            
            # recon_features = self.decoder(encoded_clip_features)
            
            
            # recon_sp_clip_features = spconv.SparseConvTensor(
            #     features=recon_features,
            #     indices= batch_dict['clip_voxel_coords'].int(),
            #     spatial_shape= np.array(self.model_cfg['CLIP_FEATURE_SHAPE']),
            #     batch_size=batch_dict['batch_size']
            # )
       
            # recon_clip_spatial_features = recon_sp_clip_features.dense()
            # N, C, D, H, W = recon_clip_spatial_features.shape
            # recon_bev_image_clip_features = recon_clip_spatial_features.view(N, C * D, H, W)       
       
            # B, C, H, W = recon_bev_image_clip_features.shape
            # recon_image_clip_features_flatten = recon_bev_image_clip_features.contiguous().flatten(2).squeeze().to(self.device)
            # recon_image_clip_features_norm = (recon_image_clip_features_flatten/(recon_image_clip_features_flatten.norm(dim=0, keepdim=True)+1e-5)).half()
            
            # recon_class_preds = (self.text_feats @ recon_image_clip_features_norm) # text_clip_features # background_index = 0 ~ 12 , object_index = 13 ~ 22
            # recon_max_score, recon_max_indices = torch.max(recon_class_preds, dim=0)
            # recon_score_masked_indices = (recon_max_score>0.02).nonzero(as_tuple=True)[0]
            
            
            # recon_height_index = (recon_score_masked_indices // W).int()
            # recon_width_index = (recon_score_masked_indices % W).int()
            # recon_spatial_infos = torch.stack((recon_height_index, recon_width_index), dim=1).cpu().detach().numpy()
            # recon_bev_vis_clip_features = np.ones((H, W, 3))
            # recon_numpy_class_idx = np.asarray(recon_max_indices[recon_score_masked_indices].cpu().detach().numpy(),dtype=int)
            # for idx in range(len(recon_numpy_class_idx)):
            #     recon_bev_vis_clip_features[recon_spatial_infos[idx,0],recon_spatial_infos[idx,1],:] = color_map[color_map_list[recon_numpy_class_idx[idx]]]
            # recon_clip_flipped_feature = torch.flip(torch.tensor(recon_bev_vis_clip_features), [1])
            # recon_clip_rotated_feature = torch.flip(torch.transpose(recon_clip_flipped_feature, 0, 1), [0]).cpu().detach().numpy()
            # plt.clf()         
            # plt.imshow(recon_clip_rotated_feature)
            # plt.savefig("/home/OpenPCDet/visualization/prediction/full/recon/with_bg_prediction_reconstruction.png",dpi=300, bbox_inches='tight')
            # plt.axis('off')
            # plt.clf() 
            # plt.close('all')
            
            # ##################################################################################################################
            
            
            
            
            
            # ########################################### prediction visualization ########################################
            
            # predicted_bev_clip_feature = batch_dict['preds_clip_bev_features']
            # N, C, H, W = predicted_bev_clip_feature.shape
            # flatten_predicted_bev_clip_feature = predicted_bev_clip_feature[0].contiguous().flatten(1).contiguous().permute(1,0)
            # decoded_flatten_predicted_bev_clip_feature = self.decoder(flatten_predicted_bev_clip_feature).contiguous().permute(1,0)
            
            # decoded_flatten_predicted_bev_clip_feature_norm = (decoded_flatten_predicted_bev_clip_feature/(decoded_flatten_predicted_bev_clip_feature.norm(dim=0, keepdim=True)+1e-5)).half()
            
            # pred_class_preds = (self.text_feats @ decoded_flatten_predicted_bev_clip_feature_norm) # text_clip_features # background_index = 0 ~ 12 , object_index = 13 ~ 22
            # pred_max_score, pred_max_indices = torch.max(pred_class_preds, dim=0)
            # pred_score_masked_indices = (pred_max_score>0.08).nonzero(as_tuple=True)[0]
            
            
            # pred_height_index = (pred_score_masked_indices // W).int()
            # pred_width_index = (pred_score_masked_indices % W).int()
            # pred_spatial_infos = torch.stack((pred_height_index, pred_width_index), dim=1).cpu().detach().numpy()
            # pred_bev_vis_clip_features = np.ones((H, W, 3))
            # pred_numpy_class_idx = np.asarray(pred_max_indices[pred_score_masked_indices].cpu().detach().numpy(),dtype=int)
            # for idx in range(len(pred_numpy_class_idx)):
            #     pred_bev_vis_clip_features[pred_spatial_infos[idx,0],pred_spatial_infos[idx,1],:] = color_map[color_map_list[pred_numpy_class_idx[idx]]]
            # pred_clip_flipped_feature = torch.flip(torch.tensor(pred_bev_vis_clip_features), [1])
            # pred_clip_rotated_feature = torch.flip(torch.transpose(pred_clip_flipped_feature, 0, 1), [0]).cpu().detach().numpy()
            # plt.clf()         
            # plt.imshow(pred_clip_rotated_feature)
            # plt.savefig("/home/OpenPCDet/visualization/prediction/full/preds/with_bg_prediction_from_centerpt_"+str(self.loss_type)+'mask'+str(self.mask)+"16_sample_1.png",dpi=300, bbox_inches='tight')
            # plt.axis('off')
            # plt.clf() 
            # plt.close('all')
            
            ##########################################################################################################

       
       
       
       
                
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            if 'clip_train' in batch_dict.keys():
                clip_loss = self.get_clip_loss(batch_dict)


                ret_dict = {
                    'loss': clip_loss
                }
                tb_dict.update({ "cliploss": clip_loss.item() })               
            else:
                ret_dict = {
                    'loss': loss 
                }

            return ret_dict, tb_dict, disp_dict
        else:
            # if batch_dict['clip_train']:
            #     self.text_feats = self.text_extractor(self.text_set, device=self.device)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    
    
    
    def get_clip_loss(self, batch_dict):
        
        preds_clip_bev_features = batch_dict['preds_clip_bev_features']
        
        # preds_sp_tensor  = spconv.SparseConvTensor.from_dense(preds_clip_bev_features)
        
        clip_features = batch_dict["clip_input_sp_tensor"].features
        encoded_clip_features = self.encoder(clip_features)
        encoded_clip_sp_tensor = spconv.SparseConvTensor(
                features=encoded_clip_features,
                indices= batch_dict['clip_voxel_coords'].int(),
                spatial_shape= np.array(self.model_cfg['CLIP_FEATURE_SHAPE']),
                batch_size=batch_dict['batch_size']
            )
        
        
        
        clip_bev_features = encoded_clip_sp_tensor.dense()
        N, C, D, H, W = clip_bev_features.shape
        clip_bev_features_reshaped = clip_bev_features.view(N, C * D, H, W)

        if self.loss_type == 'mse':
            l2loss = self.l2_loss(preds_clip_bev_features, clip_bev_features_reshaped)
            return l2loss
        
        elif self.loss_type == 'cosine':
            cosloss = self.cos_loss(preds_clip_bev_features, clip_bev_features_reshaped)
            return cosloss
        
        else:
            l2loss = self.l2_loss(preds_clip_bev_features, clip_bev_features_reshaped)
            cosloss = self.cos_loss(preds_clip_bev_features, clip_bev_features_reshaped)
            return l2loss+cosloss*0.5

        # return l2loss, cosloss

    def l2_loss(self, network_output, gt):
        
        if self.mask:
            network_output_flatten = network_output.contiguous().permute(1,0,2,3).contiguous().flatten(1)
            gt_flatten = gt.contiguous().permute(1,0,2,3).contiguous().flatten(1)
            gt_flatten_mean = gt_flatten.mean(dim=0)
            mask = (gt_flatten_mean!=0)
            masked_gt_flatten = gt_flatten[:,mask]
            masked_network_output_flatten = network_output_flatten[:,mask]

            return ((masked_network_output_flatten - masked_gt_flatten ) ** 2).mean()
        else:
             return ((network_output - gt) ** 2).mean()


    def cos_loss(self, network_output, gt):
        if self.mask:
            network_output_flatten = network_output.contiguous().permute(1,0,2,3).contiguous().flatten(1)
            gt_flatten = gt.contiguous().permute(1,0,2,3).contiguous().flatten(1)
            gt_flatten_mean = gt_flatten.mean(dim=0)
            mask = (gt_flatten_mean!=0)
            masked_gt_flatten = gt_flatten[:,mask]
            masked_network_output_flatten = network_output_flatten[:,mask]
            return 1 - F.cosine_similarity(masked_network_output_flatten, masked_gt_flatten, dim=1).mean()
        
        else:
            network_output_flatten = network_output.contiguous().permute(1,0,2,3).contiguous().flatten(1)
            gt_flatten = gt.contiguous().permute(1,0,2,3).contiguous().flatten(1)
            return 1 - F.cosine_similarity(network_output_flatten, gt_flatten, dim=1).mean()
        
    def l1_loss(self, network_output, gt):
        return torch.nn.L1Loss()(network_output, gt)
    
    
    def text_extractor(self, text_set, device):
        
        model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
        model = model.to(device)
        text_set = [f"a {c} in a scene" for c in text_set]
        text_set = clip.tokenize(text_set).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_set)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    
    def get_clip_features(self, batch_dict):
        
        pred_feats = batch_dict['encoded_spconv_tensor'].features
        pred_feats = (pred_feats/(pred_feats.norm(dim=-1, keepdim=True)+1e-5)) #dim = 128

        
        
        clip_feats = (batch_dict['clip_encoded_spconv_tensor'].features/(batch_dict['clip_encoded_spconv_tensor'].features.norm(dim=-1, keepdim=True)+1e-5)) 

        return pred_feats, clip_feats
    
    
        
    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict