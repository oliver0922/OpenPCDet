import pickle
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt


nusc = NuScenes(version='v1.0-trainval', dataroot='/home/OpenPCDet/data/nuscenes/v1.0-trainval', verbose=True)


with open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb') as f:
	full = pickle.load(f)
# print(full)




token_id = full[100]['token']
my_sample = nusc.get('sample', token_id)
nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=1, underlay_map=True, show_lidarseg=True,
				show_lidarseg_legend=True, out_path='/home/OpenPCDet/visualization/groundtruth/gt_sample_validation.jpg', axes_limit=100)
plt.close('all')
plt.clf()




partial = full[100:101]
pickle.dump(partial, open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val_vis.pkl', 'wb'))