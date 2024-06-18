import sys
import os
import time

import argparse
import yaml

from itertools import accumulate
import random

import glob
import numpy as np

import torch


from torch_scatter import scatter_min
    
class LiDomAug():
    def __init__(self, config_path, random_augmentation = False, height = 2.174):
        
        #n2k scenario
        self.random_augmentation = random_augmentation
        self.config_path = config_path

        self.vector = np.array([1,0])
        self.perturb_angle= 0
        self.perturb_dist = [0,0]
        self.height = height
    

    def init_aug_parameter(self, pose_perturb, velocity_perturb):
        if velocity_perturb:
            self.velocity = np.random.uniform(0,50/3)
            self.rotation = np.random.uniform(-np.pi/8, np.pi/8)
        else:
            self.velocity = 0
            self.rotation = 0
        
        if pose_perturb:
            self.perturb_dist = [np.random.uniform(-0.5,0.5), np.random.uniform(-1,1), np.random.uniform(-0.1,0.1)]
        else : 
            self.perturb_dist = [0,0,0]

    def initialize_config(self):
        with open(self.config_path) as file:
            config = yaml.load(file,  Loader=yaml.FullLoader)

        lidar = config['lidar']

        self.channel = int(lidar['channel'])
        self.Horizontal_res = int(lidar['Horizontal_res'])
        self.V_fov_min = np.deg2rad(float(lidar['VFoV_min']))
        self.V_fov_max = np.deg2rad(float(lidar['VFoV_max']))
        
        self.rotation_direction = int(lidar['rotation_direction'])
        self.initial_scan_degree = np.deg2rad(float(lidar['initial_scan']))

        self.rotation_rate = float(lidar['rotational_rate'])
        self.range = [lidar['range_min'],lidar['range_max']]

    def initialize_random(self):
        self.channel = 2**(int(np.random.uniform(5,7)))
        self.Horizontal_res = np.random.choice([2048])
        self.V_fov_min = np.deg2rad(np.random.uniform(-30,0))
        self.V_fov_max = np.deg2rad(np.random.uniform(0,15))

        self.rotation_direction = int(np.random.choice([-1,1]))
        self.initial_scan_degree = 0

        self.rotation_rate = 20
        self.range = [1, 120]

    def initialize(self, pose_perturb, velocity_perturb):
        self.init_aug_parameter(pose_perturb, velocity_perturb)
        if self.random_augmentation:
            self.initialize_random()
        else:
            self.initialize_config()
    
    def spherical_projection(self, temp_map, temp_label):

        map_pcd, map_label = temp_map, temp_label

        map_pcd = torch.as_tensor(map_pcd, dtype=torch.float)
        map_label = torch.as_tensor(map_label, dtype=torch.float)

        x_lidar = map_pcd[:,0] + self.perturb_dist[0]
        y_lidar = map_pcd[:,1] + self.perturb_dist[1]
        z_lidar = map_pcd[:,2] + self.perturb_dist[2]

        d_lidar = torch.sqrt(x_lidar**2+y_lidar**2+z_lidar**2)

        fov = abs(self.V_fov_max) + abs(self.V_fov_min)

        yaw = 0.5*(1-torch.atan2(y_lidar,x_lidar)/np.pi) 

        pitch = (1-(torch.asin(z_lidar/d_lidar)-self.V_fov_min)/fov)
        
        u_ = torch.as_tensor(yaw*self.Horizontal_res, dtype= torch.long)
        v_ = torch.as_tensor(pitch*self.channel, dtype= torch.long)

        ind = v_*self.Horizontal_res+u_
        
        temp_distance = np.inf*torch.ones((self.channel,self.Horizontal_res))

        V_cond = torch.logical_and(v_ >= 0, v_ < self.channel)

        D_cond = torch.logical_and(d_lidar >= self.range[0], d_lidar < self.range[1])

        Cond = torch.logical_and(V_cond,D_cond)

        temp_distance, dist_ind = scatter_min(d_lidar[Cond], ind[Cond], out=torch.flatten(temp_distance))
        temp_distance = temp_distance.reshape(self.channel, self.Horizontal_res)
        temp_distance[temp_distance == np.inf] = 0
        temp_distance = temp_distance.cpu().numpy()

        temp_labels = map_label[Cond][dist_ind-1].cpu().detach().numpy()
        
        return temp_distance.flatten(), temp_labels

    def inverse_projection(self, dist, label):

        dist = torch.from_numpy(dist)

        fov = abs(self.V_fov_max) + abs(self.V_fov_min)

        w = torch.arange(self.Horizontal_res,  dtype=torch.long)
        
        h = torch.arange(self.channel,  dtype=torch.long)

        hh, ww = torch.meshgrid(h,w)
        
        proj_w = ww/self.Horizontal_res
        
        proj_h = hh/self.channel

        yaw = (proj_w*2*(self.rotation_rate+self.rotation)/self.rotation_rate-1)*np.pi
       
        pitch = np.pi/2-(1.0*fov-proj_h*fov-abs(self.V_fov_min))

        x = dist*torch.sin(pitch)*torch.cos(-yaw)

        y = dist*torch.sin(pitch)*torch.sin(-yaw)

        z = dist*torch.cos(pitch) 

        temp_diff = self.velocity/self.rotation_rate

        x += torch.linspace(0,temp_diff*self.vector[0]*(self.Horizontal_res-1)/self.Horizontal_res,self.Horizontal_res,  dtype=torch.float64)

        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)

        point = torch.stack((x,y,z), dim=1).detach().cpu().numpy()

        label = label.flatten()

        label = np.delete(label, np.where(np.logical_or(np.isnan(point)[:,0], np.isnan(point)[:,1], np.isnan(point)[:,2])), axis=0)
        
        point = np.delete(point, np.where(np.logical_or(np.isnan(point)[:,0], np.isnan(point)[:,1], np.isnan(point)[:,2])), axis=0)

        label = np.delete(label, np.where(np.logical_and(np.abs(point[:,0]) < 0.5 , np.abs(point[:,1]) < 0.5, np.abs(point[:,2]) < 0.5)), axis=0)
        
        point = np.delete(point, np.where(np.logical_and(np.abs(point[:,0]) < 0.5 , np.abs(point[:,1]) < 0.5, np.abs(point[:,2]) < 0.5)), axis=0)

        return point, label


    def generate_frame(self, aggregate_points, pose_perturb=False, velocity_perturb=False):
   
        aggregate_points[:,2] -= self.height

        self.initialize(pose_perturb, velocity_perturb)

        aggregate_label = np.array(range(len(aggregate_points[:,0])))

        dist, label = self.spherical_projection(aggregate_points, aggregate_label)
        
        dist = dist.reshape(self.channel,self.Horizontal_res)
        label_map = label.reshape(self.channel,self.Horizontal_res)
        
        point, label = self.inverse_projection(dist, label_map)

        point[:, 2] += self.height

        return point, np.array(label, dtype=np.int64)
