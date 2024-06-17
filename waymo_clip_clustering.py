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

import open3d as o3d
import numpy as np
import time

import matplotlib.pyplot as plt
import hdbscan

import MinkowskiEngine as ME

CLIP_TEXT = ['car', 'road', 'sidewalk', 'building', 'vegetation', 'sky', 'pedestrian', 'pole', 'traffic light', 'traffic sign']
CLIP_TEXT = [f'a photo of {text}' for text in CLIP_TEXT]

clip_model, clip_preprocess = clip.load('ViT-L/14@336px', device='cpu', jit=False)
text = clip.tokenize(CLIP_TEXT)
with torch.no_grad():
    clip_text_features = clip_model.encode_text(text)

clip_text_features = clip_text_features.half().cpu().numpy()

colors_map = np.array(
    [
        [255, 158, 0],  # 1 car  orange
        [0, 0, 230],    # 2 pedestrian  Blue
        [47, 79, 79],   # 3 sign  Darkslategrey
        [220, 20, 60],  # 4 CYCLIST  Crimson
        [255, 69, 0],   # 5 traiffic_light  Orangered
        [255, 140, 0],  # 6 pole  Darkorange
        [233, 150, 70], # 7 construction_cone  Darksalmon
        [255, 61, 99],  # 8 bycycle  Red
        [112, 128, 144],# 9 motorcycle  Slategrey
        [222, 184, 135],# 10 building Burlywood
    ])

colors_map = colors_map[:,:3]


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


    
    with open('/dataset/waymo/waymo_infos_train_sampling_5.pkl', 'rb') as f:
        infos = pickle.load(f)
    

    #current_idx = 0

    
    
    for i, info in enumerate(tqdm(infos)):
        if i % 2 != 1:
            continue
        ii = i // 2 

        if ii % 16 != gpu_id:
            continue
        
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        lidar_file = DATA_PATH / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file) # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]

        openseg_file = DATA_PATH / sequence_name / ('%04d_openseg.npz' % sample_idx)
        
        openseg_feature = np.load(openseg_file)

        mask = openseg_feature['mask']

        coords, feats = ME.utils.sparse_collate(point_features[mask, :3], openseg_feature['feat'][mask])    

        pcd = o3d.io.read_point_cloud(pcd_path)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)

        # downsampling
        pcd_1 = pcd.voxel_down_sample(voxel_size=0.1)
        # remove outliers
        pcd_2, inliers = pcd_1.remove_radius_outlier(nb_points=20, radius=0.3)
        # segment plane with RANSAC
        plane_model, road_inliers = pcd_2.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=100)
        pcd_3 = pcd_2.select_by_index(road_inliers, invert=True)

        pcd_3_feat = feats[inliers][road_inliers]

        # CLUSTERING WITH HDBSCAN
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, gen_min_span_tree=True)
        #clusterer.fit(np.array(pcd_3.points))
        clusterer.fit(pcd_3_feat)
        labels = clusterer.labels_

        max_label = labels.max()
        print(f'point cloud has {max_label + 1} clusters')
        colors = plt.get_cmap("tab20")(labels / max_label if max_label > 0 else 1)
        colors[labels < 0] = 0
        pcd_3.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # generate 3D Bounding Box
        import pandas as pd
        bbox_objects = []
        indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

        MAX_POINTS = 300
        MIN_POINTS = 50

        for i in range(0, len(indexes)):
            nb_points = len(pcd_3.select_by_index(indexes[i]).points)
            if (nb_points > MIN_POINTS and nb_points < MAX_POINTS):
                sub_cloud = pcd_3.select_by_index(indexes[i])
                bbox_object = sub_cloud.get_axis_aligned_bounding_box()
                bbox_object.color = (0, 0, 1)
                bbox_objects.append(bbox_object)

        print("Number of Boundinb Box : ", len(bbox_objects))

        list_of_visuals = []
        list_of_visuals.append(pcd_3)
        list_of_visuals.extend(bbox_objects)
        # o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(f'./{sequence_name}_{sample_idx}.ply', pcd_3)