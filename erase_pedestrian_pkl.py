import pickle
import numpy as np
from tqdm import tqdm


with open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb') as f:
	full = pickle.load(f)


for i,sample in enumerate(tqdm(full)):
     gt_names = sample['gt_names']
     gt_velocity = sample['gt_boxes_velocity']
     lidar_pts = sample['num_lidar_pts']
     radar_pts = sample['num_radar_pts']
     gt_boxes = sample['gt_boxes']
     gt_tokens = sample['gt_boxes_token']
     
     num_gt = len(gt_names)
     additional_gt_names = np.array(['objectness'] * num_gt, dtype=str)
     
     gt_names = np.concatenate((gt_names,additional_gt_names),axis=0)
     gt_velocity = np.concatenate((gt_velocity, gt_velocity), axis=0)
     lidar_pts =  np.concatenate((lidar_pts, lidar_pts), axis=0)
     radar_pts =  np.concatenate((radar_pts, radar_pts), axis=0)
     gt_boxes = np.concatenate((gt_boxes, gt_boxes), axis=0)
     gt_tokens = np.concatenate((gt_tokens, gt_tokens), axis=0)
     
     # indexes = np.where(gt_names=='bus')
     # deleted_gt_names = np.delete(gt_names,indexes,axis=0)
     # deleted_gt_boxes = np.delete(gt_boxes,indexes,axis=0)
     # deleted_gt_velocity = np.delete(gt_velocity,indexes,axis=0)
     # deleted_lidar_pts = np.delete(lidar_pts,indexes,axis=0)
     # deleted_radar_pts = np.delete(radar_pts,indexes,axis=0)
     # deleted_gt_tokens = np.delete(tokens,indexes,axis=0)

     full[i]['gt_names'] = gt_names
     full[i]['gt_boxes'] = gt_boxes
     full[i]['num_lidar_pts'] = lidar_pts
     full[i]['num_radar_pts'] = radar_pts
     full[i]['gt_boxes_velocity'] = gt_velocity
     full[i]['gt_boxes_token'] = gt_tokens


pickle.dump(full, open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val_agnostic.pkl', 'wb'))
print("success!")


# partial = full[:10]
# pickle.dump(partial, open('//home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train_with_clip_ped_erased_overfit_10.pkl', 'wb'))
# print("success!")

# with open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb') as f:
# 	full = pickle.load(f)
# # print(full)

# partial = full[:1]
# pickle.dump(partial, open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'wb'))