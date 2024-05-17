import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()
        
        if "clip_voxels" in batch_dict.keys():
            clip_voxel_features, clip_voxel_num_points = batch_dict['clip_voxels'], batch_dict['clip_voxel_num_points']
            clip_voxel_features = clip_voxel_features[:,:,5:]
            clip_points_mean = clip_voxel_features[:, :, :].sum(dim=1, keepdim=False)
            clip_normalizer = torch.clamp_min(clip_voxel_num_points.view(-1, 1), min=1.0).type_as(clip_voxel_features)
            clip_points_mean = clip_points_mean / clip_normalizer
            batch_dict['clip_voxel_features'] = clip_points_mean.contiguous()

        return batch_dict
