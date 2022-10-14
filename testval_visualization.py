## Visualize the fused and seprate sequence and save the entire sequence
from dvis import dvis
import numpy as np
import struct
import yaml
from tqdm import tqdm
import os
import argparse

SEQ = ['08','11','12','13','14','15','16','17','18','19','20','21'] 

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--valtest', type=int, default='val', help='if visualize all the catagories')
    parser.add_argument('--ifstatic', type=bool, default=False, help='if visualize all the catagories')
    parser.add_argument('--iffuse', type=bool, default=False, help='if visualize all the catagories')
    parser.add_argument('--ifsave', type=bool, default=False, help='if visualize the static catagories')
    return parser.parse_args()

def unpack(compressed):
  ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
  uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
  uncompressed[::8] = compressed[:] >> 7 & 1
  uncompressed[1::8] = compressed[:] >> 6 & 1
  uncompressed[2::8] = compressed[:] >> 5 & 1
  uncompressed[3::8] = compressed[:] >> 4 & 1
  uncompressed[4::8] = compressed[:] >> 3 & 1
  uncompressed[5::8] = compressed[:] >> 2 & 1
  uncompressed[6::8] = compressed[:] >> 1 & 1
  uncompressed[7::8] = compressed[:] & 1

  return uncompressed

def read_poses(path):
    poses = []
    f = open(path)
    lines = f.readlines()
    for line in lines:
        linesplit = line.split()
        pose = np.array([[float(linesplit[0]),float(linesplit[1]),float(linesplit[2]), float(linesplit[3])],
                [float(linesplit[4]),float(linesplit[5]),float(linesplit[6]),float(linesplit[7])],
                [float(linesplit[8]),float(linesplit[9]),float(linesplit[10]),float(linesplit[11])],
                [0.0,0.0,0.0,1.0]])
        poses.append(pose)
    return poses

def read_bin(path):
    pc_list = []
    with open(path,'rb') as f:
        content = f.read()
        points = struct.iter_unpack('ffff',content)
        for point in points:
            pc_list.append([point[0], point[1],point[2]])
    return np.asarray(pc_list, dtype = np.float32)

def read_label(path):
    label_list = []
    with open(path,'rb') as f:
        content = f.read()
        labels = struct.iter_unpack('I',content)
        for label in labels:
            label_list.append(label[0])
    return label_list

def idx2str(idx):
    if idx<10:
        return '00000'+str(idx)
    elif idx<100:
        return '0000'+str(idx)
    elif idx<1000:
        return '000'+str(idx)
    else:
        return '00'+str(idx)

def dot(transform, pts):
    if pts.shape[1] == 3:
        pts = np.concatenate([pts,np.ones((len(pts),1))],1)
    return (transform @ pts.T).T

def main():
    args = parse_args()
    for seq in SEQ[:1]:
        velo2cam = np.array([[0,-1,0,0],[0,0,-1,-0.08],[1,0,0,-0.27],[0,0,0,1]])
        color_dict = yaml.safe_load(open('ws_hiwi/semantic-kitti.yaml', 'r'))["color_map"]
        if args.valtest == 'test':
            path = 'JS3C-Net/log/JS3C-Net-kitti/dump/completion/submit_test2022_09_04/sequences/'+ seq +'/predictions/' #test 
        else:
            path = 'JS3C-Net/log/JS3C-Net-kitti/dump/completion/submit_valid2022_08_18/sequences/'+ seq +'/predictions/' #val
        path_poses = 'dataset/sequences/'+ seq +'/poses.txt'
        poses = read_poses(path_poses)
        volume_colors = np.array([[0,0,0,0,0,0]])
        save_array = np.array([[0,0,0,0]])
        poses_save = np.zeros([1,17])
        startframe = 0
        endframe = len(poses)
        for i in tqdm(range(startframe, endframe,5)):
            path_labels = path + idx2str(i) +'.label'
            if os.path.exists(path_labels):
                pose_velo = np.linalg.inv(velo2cam).dot(poses[i].dot(velo2cam))
                pose_flat = pose_velo.reshape(1,16)
                pose_flat = np.concatenate([np.array([[i]]), pose_flat],1)
                poses_save = np.concatenate([poses_save, pose_flat],0)
                labels_out = np.fromfile(path_labels, dtype=np.uint16).reshape(256,256,64)[:128,:128,:32]
                #upsampling
                labels = np.zeros([256,256,32])
                labels[::2,::2,:] = labels_out
                labels[1::2,::2,:] = labels_out
                labels[::2,1::2,:] = labels_out
                labels[1::2,1::2,:] = labels_out
                if args.ifstatic:
                    valid_label_mask = (labels>39) & (labels <100) #extract the static thing
                else:
                    valid_label_mask = labels>0 # all label
                valid_labels = labels[valid_label_mask]
                valid_label_inds = np.stack(np.nonzero(valid_label_mask),1)

                vox2scene = np.eye(4)
                vox2scene[:3,:3] = np.diag([1/5,1/5,1/5])
                vox2scene[:3,3] = np.array([0.1,-25.5,-1.9])
                valid_scene_coords = dot(vox2scene, valid_label_inds)
                valid_scene_coords_global = valid_scene_coords.dot(pose_velo.T)[:,:3]

                valid_colors = np.zeros((len(valid_scene_coords),3))
                for label in np.unique(valid_labels):
                    if label>0:
                        label_mask = valid_labels == label
                        valid_colors[label_mask] = color_dict[label]
            
                # convert to aligned voxel
                valid_scene_coords_global = valid_scene_coords_global*5
                valid_scene_coords_global.astype(int)
                valid_scene_coords_global = valid_scene_coords_global/5
                valid_scene_coords_col = np.concatenate([valid_scene_coords_global,valid_colors],1)

                valid_labels = np.array([valid_labels]).T
                xyzlabel = np.concatenate([valid_scene_coords_global, valid_labels],1)
                if not args.iffuse:
                    dvis(valid_scene_coords_col, l=5, t=i, vs=1/5, name='prediction/semantic volume'+ str(i))
                if args.iffuse:  
                    volume_colors = np.concatenate([volume_colors, valid_scene_coords_col],0)
                if args.ifsave:
                    save_array = np.concatenate([save_array, xyzlabel],0)

        if args.iffuse:
            volume_colors = np.unique(volume_colors, axis=0)# delete the repeated voxel
            dvis(volume_colors, l=4, vs=1/5, ms=1000000, name='volume/fused volume')
        if args.ifsave:
            save_array = np.unique(save_array, axis=0)# delete the repeated voxel
            np.save('ws_hiwi/seq'+seq+'_alllabel.npy',save_array)
            np.save('ws_hiwi/seq'+seq+'_pose.npy',poses_save[1:])
    

if __name__=="__main__":
    main()