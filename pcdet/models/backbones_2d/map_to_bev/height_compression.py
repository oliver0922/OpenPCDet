import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if "encoded_spconv_tensor" in batch_dict.keys():
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            spatial_features = encoded_spconv_tensor.dense()
            N, C, D, H, W = spatial_features.shape
            spatial_features = spatial_features.view(N, C * D, H, W)
            batch_dict['spatial_features'] = spatial_features
            batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        
        
        # if self.training and "clip_input_sp_tensor" in batch_dict.keys() :
        #     clip_encoded_spconv_tensor = batch_dict["clip_input_sp_tensor"]
        #     clip_spatial_features = clip_encoded_spconv_tensor.dense()
        #     N, C, D, H, W = clip_spatial_features.shape
        #     clip_spatial_features = clip_spatial_features.view(N, C * D, H, W)
        #     batch_dict['clip_bev_features'] = clip_spatial_features
        
        
        return batch_dict
