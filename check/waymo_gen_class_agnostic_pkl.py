import pickle
import numpy as np
import tqdm

with open('/home/OpenPCDet/data/waymo/waymo_infos_train_sampling_5.pkl', 'rb') as f:
	full = pickle.load(f)
 
 
for i in range(len(full)):
    print(i)
    
    
    
    part = full[i]
    part_anno = part['annos'] 
    
    
    name = part_anno['name']
    difficulty = part_anno['difficulty']
    dimensions = part_anno['dimensions']  
    location = part_anno['location']
    heading_angles = part_anno['heading_angles']
    obj_ids= part_anno['obj_ids']
    tracking_difficulty = part_anno['tracking_difficulty']
    num_points_in_gt = part_anno['num_points_in_gt']
    speed_global = part_anno['speed_global']
    accel_global = part_anno['accel_global']
    gt_boxes_lidar = part_anno['gt_boxes_lidar']    


    variables = [
    name, difficulty, dimensions, location, heading_angles, 
    obj_ids, tracking_difficulty, num_points_in_gt, 
    speed_global, accel_global, gt_boxes_lidar
    ]

    first_dim = variables[0].shape[0]
    for var in variables:
        if var.shape[0] != first_dim:
            print("different!")

    num_gt = len(name)
    additional_gt_names = np.array(['objectness'] * num_gt, dtype=str)

    name = np.concatenate((name, additional_gt_names),axis=0)
    difficulty  = np.concatenate((difficulty,difficulty ), axis=0)
    dimensions = np.concatenate((dimensions,dimensions), axis=0)
    location = np.concatenate((location,location), axis=0)
    heading_angles = np.concatenate((heading_angles,heading_angles), axis=0)    
    obj_ids = np.concatenate((obj_ids,obj_ids), axis=0)    
    tracking_difficulty = np.concatenate((tracking_difficulty,tracking_difficulty), axis=0)    
    num_points_in_gt = np.concatenate((num_points_in_gt,num_points_in_gt), axis=0)    
    speed_global = np.concatenate((speed_global,speed_global), axis=0)    
    accel_global = np.concatenate((accel_global,accel_global), axis=0)
    gt_boxes_lidar = np.concatenate((gt_boxes_lidar,gt_boxes_lidar), axis=0)


    new_annos = dict()
    new_annos.update({'name':name,
                 'difficulty':difficulty,
                 'dimensions':dimensions,
                 'location': location,
                 'heading_angles': heading_angles,
                 'obj_ids': obj_ids,
                 'tracking_difficulty': tracking_difficulty,
                 'num_points_in_gt': num_points_in_gt,
                 'speed_global': speed_global,
                 'accel_global': accel_global,
                 'gt_boxes_lidar':gt_boxes_lidar                
                 })
    
    full[i]['annos'] = new_annos
    
pickle.dump(full, open('/home/OpenPCDet/data/waymo/waymo_infos_train_sampling_5_agnostic.pkl', 'wb'))