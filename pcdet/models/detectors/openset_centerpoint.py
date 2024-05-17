from .detector3d_template import Detector3DTemplate
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.clip_spconv_backbone import Clip_VoxelResBackBone8x
from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE
from ...utils.spconv_utils import  spconv
import numpy as np



class Openset_CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.dataset = dataset
        self.logger = dataset.logger
        if 'COS_LOSS_WEIGHT' in self.model_cfg.keys():
            self.cos_loss_weight = self.model_cfg.COS_LOSS_WEIGHT
        else:
            self.cos_loss_weight = 1
        
        self.predefined_label = ['car','truck', 'construction_vehicle','trailer',
              'barrier', 'motorcycle', 'bicycle', 'traffic_cone']
    
        self.text_set = ['pedestrian','barrier', 'barricade', 'bicycle', 'bus','car', 'bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck',
                'motorcycle', 'person','traffic cone', 'trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container',
                'truck', 'road', 'curb', 'traffic island', 'traffic median', 'sidewalk', 'grass', 'grassland', 'lawn', 'meadow', 'turf', 'sod',
                'building', 'wall', 'pole', 'awning', 'tree', 'trunk', 'tree trunk', 'bush', 'shrub', 'plant', 'flower', 'woods', 'police_officer']
        
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
        


    def forward(self, batch_dict):
        
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)


        
        ####### only clip_feature_training ######################
        if batch_dict['clip_train']:
            
            
            
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
            for clip_param in self.dense_head.heads_list[6].clip.parameters():
                clip_param.requires_grad = True
                
            # freezed_modules = [] ####### 학습 가능한 layer 찾기
            # for name, param in self.named_parameters():
            #     if param.requires_grad:
            #         freezed_modules.append(name)    
            
                
        ###################################################################################                       
       
       
       
       
       
       
       
                
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            if batch_dict['clip_train']:
                # l2loss, cosloss = self.get_clip_loss(batch_dict)
                cosloss = self.get_clip_loss(batch_dict)


                ret_dict = {
                    'loss': cosloss 
                }
                # tb_dict.update({ "l2loss": l2loss.item(), "cosloss": cosloss.item() })
                tb_dict.update({ "cosloss": cosloss.item() })               
            else:
                ret_dict = {
                    'loss': loss 
                }

            return ret_dict, tb_dict, disp_dict
        else:
            if batch_dict['clip_train']:
                self.text_feats = self.text_extractor(self.text_set, device=self.device)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    
    def get_clip_loss(self, batch_dict):
        
        point_clip_feature = batch_dict['preds_clip_bev_features']
        # pred_feats = (point_clip_feature/(point_clip_feature.norm(dim=1, keepdim=True)+1e-5))
        # clip_feats = (batch_dict['clip_bev_features']/(batch_dict['clip_bev_features'].norm(dim=1, keepdim=True)+1e-5))
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

        
        # l2loss = self.l2_loss(pred_feats, clip_feats)
        cosloss = self.cos_loss(point_clip_feature, clip_bev_features_reshaped)
        return cosloss

        # return l2loss, cosloss

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