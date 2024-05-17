import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_1 = torch.load("/home/OpenPCDet/output/cfgs/openset/train/centerpoint-nuscenes_detection_train_clip_freeze/default/ckpt/checkpoint_epoch_30.pth", map_location=device)
# model_2 = torch.load("/home/OpenPCDet/output/cfgs/openset/train/centerpoint-nuscenes_detection_freeze_clip_train/default/ckpt/latest_model.pth", map_location=device)
# model_3 = torch.load("/home/OpenPCDet/output/cfgs/openset/train/centerpoint-nuscenes_detection_freeze_clip_train/default/ckpt/checkpoint_epoch_1.pth", map_location=device)
# print("success!")

state_dict1 = torch.load('/home/OpenPCDet/output/cfgs/openset/train/centerpoint-nuscenes_detection_train_clip_freeze/default/ckpt/checkpoint_epoch_30.pth')
state_dict2 = torch.load('/home/OpenPCDet/output/cfgs/openset/centerpoint-nuscenes_reconstruction/default/ckpt/latest_model.pth')

# 빈 딕셔너리 생성
combined_state_dict = {}

combined_state_dict.update(state_dict1)
combined_state_dict['model_state'].update(state_dict2['model_state'])


torch.save(combined_state_dict, '/home/OpenPCDet/output/combined_weight_detection30_reconlatest.pth')