import numpy as np
import pickle
from pathlib import Path

import math

import tensorflow as tf2
import tensorflow.compat.v1 as tf
from tensorflow import io
from tqdm import tqdm
import torch
import argparse

from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils as fu
import os

import cv2
from PIL import Image
import clip 

from lidomaug import LiDomAug

WAYMO_DATA_PATH = Path('/dataset/waymo')
DATA_PATH = Path('/dataset/waymo/waymo_processed_data_v0_5_0')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of OpenSeg on Waymo.')
    parser.add_argument('--run_device', type=int, help='multiframe_number', default=0)

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    
    gpu_id = args.run_device
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    Augmentation = LiDomAug('./lidomaug_config/LiDAR_config_v64.yaml')


    with open('/dataset/waymo/waymo_infos_train_sampling_5.pkl', 'rb') as f:
        infos = pickle.load(f)

    current_tfrecord = ''
    #current_idx = 0
    poses = []
    pts = []
    cnt = 0

    sample_min = -20
    sample_max = 20

    Augmentation.initialize_config()

    for i, info in enumerate(tqdm(infos)):

        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        pose_file = DATA_PATH / sequence_name / 'poses.npy'
        poses = np.load(pose_file)

        pts = []
        pts_feat = []

        for index in range(sample_idx+sample_min, sample_idx+sample_max):
            lidar_file = DATA_PATH / sequence_name / ('%04d.npy' % index)
            if not os.path.exists(lidar_file):
                continue
            pose = poses[index]
            points = np.load(lidar_file)
            points = points[:, :3]
            point_features = points[:,3:]
            points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
            points = np.matmul(points, pose.T)
            pts.append(points) 
            pts_feat.append(point_features)

        pts = np.concatenate(pts, axis=0)
        pts_feat = np.concatenate(pts_feat, axis=0)
        pts = np.matmul(pts, np.linalg.inv(poses[sample_idx]).T)
        pts[:, 2] -= 2.184

        pts, index = Augmentation.generate_frame(pts[:,:3], False, False)
        pts[:, 2] += 2.184
        #import pdb; pdb.set_trace()
        pts_feat = pts_feat[index,:]

        import open3d as o3d 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
        o3d.io.write_point_cloud('test.ply', pcd)

        import pdb; pdb.set_trace()

    import o3d.geometry.PointCloud as pcd
    #np.save(pose_file, poses)