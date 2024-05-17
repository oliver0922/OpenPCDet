import torch

model = torch.load("/home/OpenPCDet/output/cfgs/openset/train/centerpoint-nuscenes/default/ckpt_10_class/checkpoint_epoch_1.pth")
print(model)
