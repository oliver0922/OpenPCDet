from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.splits import create_splits_scenes
import os

# nuScenes 데이터셋 로드
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/OpenPCDet/data/nuscenes/v1.0-trainval', verbose=True)

def find_cam_front_from_cam_front_right(cam_front_right_filename):
    # 해당 파일의 샘플 데이터를 찾기 위해 파일 이름의 토큰 추출
    cam_front_right_sample_data = nusc.get('sample_data', cam_front_right_filename)
    
    # 해당 샘플 데이터를 포함하는 샘플을 로드
    sample = nusc.get('sample', cam_front_right_sample_data['sample_token'])
    
    # 해당 샘플의 'CAM_FRONT'에 대한 샘플 데이터 토큰을 로드
    cam_front_token = sample['data']['CAM_FRONT']
    
    # 'CAM_FRONT'에 대한 샘플 데이터 로드
    cam_front_sample_data = nusc.get('sample_data', cam_front_token)
    
    # 파일 이름 반환
    return cam_front_sample_data['filename']

# 'cam_front_right' 파일 이름 예제
cam_front_right_filename = 'samples/CAM_FRONT_RIGHT/n008-2018-08-31-11-56-46-0400__CAM_FRONT_RIGHT__1535731382670482.jpg'


# 'cam_front_right' 파일에 해당하는 'cam_front' 파일 찾기
cam_front_filename = find_cam_front_from_cam_front_right(cam_front_right_filename)
print(f'The corresponding CAM_FRONT file is: {cam_front_filename}')