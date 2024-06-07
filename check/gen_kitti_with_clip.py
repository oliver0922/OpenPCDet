import pickle

with open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb') as f:
	full = pickle.load(f)
 
 
 
for part in full:
     clip_feature_path = '/home/OpenPCDet/data/kitti' '/nuScene_lidarseg_split/train/scene-0001/openseg/1531883530449377.npz'
     part.update({"clip_feature_path":clip_feature_path})
 
 
pickle.dump(full, open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val_vis.pkl', 'wb'))