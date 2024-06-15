import pickle
import numpy as np
import tqdm

with open('/home/OpenPCDet/data/kitti/kitti_infos_trainval_include_dontcare.pkl', 'rb') as f:
	full = pickle.load(f)
 
print("1")

for i in range(len(full)):
    print(i)
    
    
    
    part = full[i]
    part_anno = part['annos']
    
    name = part_anno['name']
    
    if 'Van' in name:
        print("1")
    
    dontcare_mask = (name != 'DontCare')
    masked_name = name[dontcare_mask]
    truncated = part_anno['truncated'][dontcare_mask] 
    occluded = part_anno['occluded'][dontcare_mask] 
    alpha = part_anno['alpha'][dontcare_mask]  
    bbox = part_anno['bbox'][dontcare_mask] 
    dimensions = part_anno['dimensions'][dontcare_mask]   
    location = part_anno['location'][dontcare_mask] 
    rotation_y = part_anno['rotation_y'][dontcare_mask]  
    score = part_anno['score'][dontcare_mask] 
    difficulty = part_anno['difficulty'][dontcare_mask]      
    index = part_anno['index'][dontcare_mask] 
    gt_boxes_lidar = part_anno['gt_boxes_lidar'][dontcare_mask]      
    num_points_in_gt = part_anno['num_points_in_gt'][dontcare_mask]   
    
    num_gt = len(masked_name)
    additional_gt_names = np.array(['objectness'] * num_gt, dtype=str)
    
    
    name = np.concatenate((masked_name, additional_gt_names),axis=0)
    truncated = np.concatenate((truncated,truncated), axis=0)
    occluded = np.concatenate((occluded,occluded), axis=0)
    alpha = np.concatenate((alpha,alpha), axis=0)
    bbox = np.concatenate((bbox,bbox), axis=0)
    dimensions = np.concatenate((dimensions,dimensions), axis=0)
    location = np.concatenate((location,location), axis=0)
    rotation_y = np.concatenate((rotation_y,rotation_y), axis=0)
    score = np.concatenate((score,score), axis=0)   
    difficulty  = np.concatenate((difficulty,difficulty ), axis=0)
    index = np.concatenate((index,index), axis=0) 
    gt_boxes_lidar = np.concatenate((gt_boxes_lidar,gt_boxes_lidar), axis=0)
    num_points_in_gt = np.concatenate((num_points_in_gt,num_points_in_gt), axis=0)
    
    new_annos = dict()
    new_annos.update({'name':name,
                 'truncated':truncated,
                 'occluded':occluded,
                 'alpha':alpha,
                 'bbox':bbox,
                 'dimensions':dimensions,
                 'rotation_y':rotation_y,
                 'score':score,
                 'difficulty':difficulty,
                 'index':index,
                 'gt_boxes_lidar':gt_boxes_lidar,
                 'num_points_in_gt': num_points_in_gt,
                 'location': location                 
                 })
    
    full[i]['annos'] = new_annos
                  
pickle.dump(full, open('/home/OpenPCDet/data/kitti/kitti_infos_trainval_agnostic_include_dontcare.pkl', 'wb'))