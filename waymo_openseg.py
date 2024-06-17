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

def read_bytes(path):
    '''Read bytes for OpenSeg model running.'''

    with io.gfile.GFile(path, 'rb') as f:
        file_bytes = f.read()
    return file_bytes


def load_calib(calibration, image):
    extrinsic = tf.reshape(
        tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
        [4, 4],
    )
    intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
    metadata = tf.constant(
        [
            calibration.width,
            calibration.height,
            calibration.rolling_shutter_direction,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(image.pose.transform) + [
        image.velocity.v_x,
        image.velocity.v_y,
        image.velocity.v_z,
        image.velocity.w_x,
        image.velocity.w_y,
        image.velocity.w_z,
        image.pose_timestamp,
        image.shutter,
        image.camera_trigger_time,
        image.camera_readout_done_time,
    ]
    return [extrinsic, intrinsic, metadata, camera_image_metadata]


def extract_openseg_img_feature(img_path, openseg_model, regional_pool=True):
    '''Extract per-pixel OpenSeg features.'''

    text_emb = tf.zeros([1, 1, 768])
    # load RGB image
    #np_image_string = bytearray(img)
    np_image_string = read_bytes(img_path)
    # run OpenSeg
    results = openseg_model.signatures['serving_default'](inp_image_bytes= tf.convert_to_tensor(np_image_string), inp_text_emb=text_emb)
    
    img_info = results['image_info']
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1])
    ]
    if regional_pool:
        image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
    else:
        image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]

    feat_2d = tf.cast(image_embedding_feat[[0]], dtype=tf.float16).numpy()
    


    del results
    del image_embedding_feat
    feat_2d = torch.from_numpy(feat_2d).permute(2, 0, 1) # dtype=torch.float16
    
    return feat_2d

def compute_mapping(world_to_camera, coords, depth=None, intrinsic=None, image_dim=(1920, 1280)):
    """
    :param camera_to_world: 4 x 4
    :param coords: N x 3 format
    :param depth: H x W format
    :param intrinsic: 3x3 format
    :return: mapping, N x 3 format, (H,W,mask)
    """

    mapping = np.zeros((3, coords.shape[0]), dtype=int)
    coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
    assert coords_new.shape[0] == 4, "[!] Shape error"

    #world_to_camera = np.linalg.inv(camera_to_world)
    p = np.matmul(world_to_camera, coords_new)
    p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
    p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
    pi = np.round(p).astype(int) # simply round the projected coordinates
    inside_mask = (pi[0] >= 5) * (pi[1] >= 5) \
                * (pi[0] < image_dim[0]-5) \
                * (pi[1] < image_dim[1]-5)
    if depth is not None:
        depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
        occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                - p[2][inside_mask]) <= \
                                0.25 * depth_cur

        inside_mask[inside_mask == True] = occlusion_mask
    else:
        front_mask = p[2]>0 # make sure the depth is in front
        inside_mask = front_mask*inside_mask
    mapping[0][inside_mask] = pi[1][inside_mask]
    mapping[1][inside_mask] = pi[0][inside_mask]
    mapping[2][inside_mask] = 1

    return mapping.T


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''
    intrinsic_0 = np.eye(3)

    intrinsic_0[0, 0] = intrinsic[0]
    intrinsic_0[1, 1] = intrinsic[1]
    intrinsic_0[0, 2] = intrinsic[2]
    intrinsic_0[1, 2] = intrinsic[3]
    intrinsic = intrinsic_0

    #intrinsic = np.array(intrinsic).reshape(3,3)

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation





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
    
    openseg_model =  tf2.saved_model.load('/root/code/openseg_exported_clip', tags=[tf.saved_model.tag_constants.SERVING],)


    current_tfrecord = ''
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
        
        '''
        if (int(sample_idx) == 167) & (sequence_name == 'segment-4487677815262010875_4940_000_4960_000_with_camera_labels'):
            flag = True
        else : 
            flag = False
            continue
        '''
        flag = False

        points_all = point_features[:, :3]

        tf_record_path = WAYMO_DATA_PATH / 'training' / f'{sequence_name}.tfrecord'
        
        if tf_record_path != current_tfrecord:
            current_tfrecord = tf_record_path
            dataset = tf.data.TFRecordDataset(tf_record_path, compression_type='')

        for raw_record in dataset.skip(sample_idx).take(1):
            data = raw_record
        
        
        frame = dataset_pb2.Frame()
        frame.ParseFromString(data.numpy())

        ret_outputs = frame_utils.parse_range_image_and_camera_projection(frame)

        if len(ret_outputs) == 4:
            range_images, camera_projections, seg_labels, range_image_top_pose = ret_outputs
        else:
            assert len(ret_outputs) == 3
            range_images, camera_projections, range_image_top_pose = ret_outputs

        use_two_returns = True 

        points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1) if use_two_returns else (0,)
        )


        points = np.concatenate(points, axis=0)
        cp_points = np.concatenate(cp_points, axis=0)

        assert len(points_all) == len(points)

        mask_full = cp_points[..., 0] != 0

        feat_3d = np.zeros((len(points), 768), dtype=np.float16)

        #color = np.zeros((len(points), 3), dtype=np.uint8)

        count = np.zeros((len(points), 1), dtype=np.int8)

        for j in range(5):

            img = frame.images[j]

            projection = cp_points[cp_points[..., 0] == img.name]

            numpy_image = tf.image.decode_jpeg(img.image).numpy()

            img_size = numpy_image.shape[0:2]

            if img_size[0] == 1280:
                crop_type = 0
            else : 
                crop_type = 1

            np_img_list = []
            clip_feature_list = []
 
            
            #np_img_list.append(numpy_image[:, :img_size[1]//2, :])
            #np_img_list.append(numpy_image[:, img_size[1]//2:, :])
            if crop_type == 0:
                np_img_list.append(numpy_image[:img_size[0]//2, :img_size[1]//2, :])
                np_img_list.append(numpy_image[img_size[0]//2:, :img_size[1]//2, :])
                np_img_list.append(numpy_image[:img_size[0]//2, img_size[1]//2:, :])
                np_img_list.append(numpy_image[img_size[0]//2:, img_size[1]//2:, :])

            else:
                np_img_list.append(numpy_image[:, :img_size[1]//2, :])
                np_img_list.append(numpy_image[:, img_size[1]//2:, :])


            clip_feature_list = []
            for k, np_img in enumerate(np_img_list):
                PIL_image = Image.fromarray(np_img)
                #resize_img_size = [427, 640]
                resize_img_size = [427, 640]
                PIL_image = PIL_image.resize((resize_img_size[1], resize_img_size[0]))
                PIL_image.save(f'temp_{gpu_id}_{j}_{k}.png')
                clip_feature = extract_openseg_img_feature(f'temp_{gpu_id}_{j}_{k}.png', openseg_model, regional_pool=True)
                clip_feature_list.append(clip_feature)

            resize_img_size = [clip_feature_list[0].shape[1], clip_feature_list[0].shape[2]]
            if crop_type == 0:
                clip_feature_2d = torch.zeros_like(clip_feature_list[0]).repeat(1, 2, 2)
                clip_feature_2d[:, :resize_img_size[0], :resize_img_size[1]] = clip_feature_list[0]
                clip_feature_2d[:, resize_img_size[0]:, :resize_img_size[1]] = clip_feature_list[1]
                clip_feature_2d[:, :resize_img_size[0], resize_img_size[1]:] = clip_feature_list[2]
                clip_feature_2d[:, resize_img_size[0]:, resize_img_size[1]:] = clip_feature_list[3]
            else:
                clip_feature_2d = torch.zeros_like(clip_feature_list[0]).repeat(1, 1, 2)
                clip_feature_2d[:, :, :resize_img_size[1]] = clip_feature_list[0]
                clip_feature_2d[:, :, resize_img_size[1]:] = clip_feature_list[1]

            resize_img_size = [clip_feature_2d.shape[1],clip_feature_2d.shape[2]]
            resized_projection = np.zeros((len(projection),2), dtype=np.int16)
            resized_projection[:,0] = projection[:,2] / img_size[0] * resize_img_size[0]
            resized_projection[:,1] = projection[:,1] / img_size[1] * resize_img_size[1]
            
            #resized_projection[:,0] = 2 * (projection[:,2] / img_size[0]) - 1 
            #resized_projection[:,1] = 2 * (projection[:,1] / img_size[1]) - 1 

            #resized_projection = torch.from_numpy(resized_projection).unsqueeze(0).unsqueeze(0).float()
            #clip_feature_2d = clip_feature_2d.unsqueeze(0).float()

            #clip_feature_2d = torch.nn.functional.grid_sample(clip_feature_2d, resized_projection, mode='bilinear', padding_mode='border', align_corners=True)
            
            feat_3d[cp_points[..., 0] == img.name] += clip_feature_2d[:, resized_projection[:,0], resized_projection[:,1]].half().squeeze().T.numpy()
            count[cp_points[..., 0] == img.name] += 1

        count[count == 0] = 1e-5
        feat_3d /= count    
        
        if flag:
            
            feat_3d = feat_3d / (np.linalg.norm(feat_3d, axis=1, keepdims=True) + 1e-5)
            clip_text_features = clip_text_features / (np.linalg.norm(clip_text_features, axis=1, keepdims=True) + 1e-5)

            clip_score = feat_3d @ clip_text_features.T
            clip_logit = torch.nn.functional.softmax(torch.from_numpy(clip_score).float(), dim=1).numpy()
            clip_pred = np.argmax(clip_logit, axis=1)
            clip_pred = clip_pred.squeeze()
            clip_pred = clip_pred.astype(np.uint8)
            color = colors_map[clip_pred]
            
            #color /= count
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_all)
            pcd.colors = o3d.utility.Vector3dVector(color / 255)
            o3d.io.write_point_cloud(f'point_cloud_2.ply', pcd)


        openseg_file = lidar_file = DATA_PATH / sequence_name / ('%04d_openseg.npz' % sample_idx)

        np.savez_compressed(openseg_file, feat = feat_3d, mask_full = mask_full)

