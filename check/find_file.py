from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.splits import create_splits_scenes
import os
import pickle
import matplotlib.pyplot as plt


# nuScenes 데이터셋 로드
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/OpenPCDet/data/nuscenes/v1.0-trainval', verbose=True)

with open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb') as f:
	full = pickle.load(f)

cam_channel = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK_RIGHT','CAM_BACK_LEFT','CAM_BACK']

cam_name = 'n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984244404844.jpg'

for i,sample in enumerate(nusc.sample):
    # sample 데이터에서 camera_channel에 해당하는 데이터 가져오기
    for channel in cam_channel:
        sample_data_token = sample['data'][channel]
        sample_data = nusc.get('sample_data', sample_data_token)
    
    # sample_data의 파일 이름이 주어진 파일 이름과 일치하는지 확인
        if os.path.basename(sample_data['filename']) == cam_name:
            print(sample['token'])
            token_name = sample['token']
for partial in full:
    if partial['token'] == token_name:
        print("success")
        my_sample = nusc.get('sample', token_name)
        nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=1, underlay_map=True, show_lidarseg=True,
			show_lidarseg_legend=True, out_path='/home/OpenPCDet/visualization/groundtruth/gt_sample_validation_LIDAR.jpg', axes_limit=100)
        plt.close('all')
        plt.clf()
        for channel in cam_channel: 
            nusc.render_sample_data(my_sample['data'][channel], nsweeps=1, underlay_map=True, show_lidarseg=True,
				show_lidarseg_legend=True, out_path='/home/OpenPCDet/visualization/groundtruth/gt_sample_validation'+str(channel)+'.jpg', axes_limit=100)
            plt.close('all')
            plt.clf()
        
partial_list = list()
partial_list.append(partial)
pickle.dump(partial_list, open('/home/OpenPCDet/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val_with_clip_suv.pkl', 'wb'))   