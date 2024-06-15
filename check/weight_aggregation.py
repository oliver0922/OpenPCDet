import torch


####### nuscenes ############
# state_dict1 = torch.load('/home/OpenPCDet/output/cfgs/openset/train/centerpoint-nuscenes_detection_train_clip_freeze/default/ckpt/checkpoint_epoch_30.pth')
# state_dict2 = torch.load('/home/OpenPCDet/output/cfgs/openset/centerpoint-nuscenes_reconstruction/default/ckpt/latest_model.pth')

# combined_state_dict = {}

# combined_state_dict.update(state_dict1)
# combined_state_dict['model_state'].update(state_dict2['model_state'])


# torch.save(combined_state_dict, '/home/OpenPCDet/output/combined_weight_detection30_reconlatest.pth')
##############################


####### kitti ############
#detection
state_dict1_kitti = torch.load('/home/OpenPCDet/output/cfgs/hyundai/exp1/model/openset/openset_kitti_train_waymo_test_detection_train_clip_freeze/default/ckpt/checkpoint_epoch_80.pth')
#reconstruction
state_dict2_kitti = torch.load('/home/OpenPCDet/output/cfgs/hyundai/exp1/model/autoencoder/kitti_clip_autoencoder/default/ckpt/latest_model.pth')

combined_state_dict_kitti = {}

combined_state_dict_kitti.update(state_dict1_kitti)
combined_state_dict_kitti['model_state'].update(state_dict2_kitti['model_state'])


torch.save(combined_state_dict_kitti, '/home/OpenPCDet/output/kitti_combined_weight_detection80_reconlatest.pth')
##############################