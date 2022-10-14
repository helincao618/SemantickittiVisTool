from dvis import dvis
import numpy as np
from tqdm import tqdm
import os
import argparse

SEQ = ['08','11','12','13','14','15','16','17','18','19','20','21']

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
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
    velo2cam = np.array([[0,-1,0,0],[0,0,-1,-0.08],[1,0,0,-0.27],[0,0,0,1]])
    for seq in SEQ:
        path = 'dataset/sequences/'+ seq +'/voxels/'
        path_poses = 'dataset/sequences/'+ seq +'/poses.txt'
        poses = read_poses(path_poses)
        volume = np.array([[0,0,0]])
        save_array = np.array([[0,0,0,0]])

        startframe = 0
        endframe = len(poses)
        for i in tqdm(range(startframe, endframe,5)):
            path_input = path + idx2str(i) +'.bin'
            if os.path.exists(path_input):
                pose_velo = np.linalg.inv(velo2cam).dot(poses[i].dot(velo2cam))
                labels = unpack(np.fromfile(path_input, dtype=np.uint8)).reshape(256,256,32)
                valid_label_mask = labels>0 #extract the static thing
                valid_labels = labels[valid_label_mask]
                valid_label_inds = np.stack(np.nonzero(valid_label_mask),1)
                vox2scene = np.eye(4)
                vox2scene[:3,:3] = np.diag([1/5,1/5,1/5])
                vox2scene[:3,3] = np.array([0.1,-25.5,-1.9])
                valid_scene_coords = dot(vox2scene, valid_label_inds)
                valid_scene_coords_global = valid_scene_coords.dot(pose_velo.T)[:,:3]
                valid_labels = np.array([valid_labels]).T
                idxxyz = np.concatenate([np.ones([len(valid_scene_coords_global),1])*i,valid_scene_coords_global],1)
                if not args.iffuse:
                    dvis(valid_scene_coords_global, l=1, t=i, vs=1/5, name='input/volume'+ str(i))

                if args.iffuse:
                    volume = np.concatenate([volume, valid_scene_coords_global],0)
                if args.ifsave:
                    save_array = np.concatenate([save_array, idxxyz],0)

        if args.iffuse:
            dvis(volume, l=2, vs=1/5, ms=1000000, name='volume/fused volume')
        if args.ifsave:
            np.save('ws_hiwi/seq'+seq+'_input.npy',save_array)

if __name__=="__main__":
    main()