## Visualize the training data, including point clouds with texture/ points clouds with sematic label/ 
## groundtruth volume/ and fused result for all above

from dvis import dvis
import numpy as np
import struct
import yaml
from tqdm import tqdm
from matplotlib import image
import argparse

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--iffusion', type=bool, default=False, help='if the data need to be fused')
    parser.add_argument('--startframe', type=int, default=0, help='Start frame to visualization')
    parser.add_argument('--endframe', type=int, default=10, help='End frame to visualization')
    parser.add_argument('--seqpath', type=str, default='dataset/sequences/00/', help='choose the path of the sequence to visualize')
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

def read_calib(path):
    calibs = []
    f = open(path)
    lines = f.readlines()
    for line in lines:
        linesplit = line[4:].split()
        calib = np.array([[float(linesplit[0]),float(linesplit[1]),float(linesplit[2]), float(linesplit[3])],
                [float(linesplit[4]),float(linesplit[5]),float(linesplit[6]),float(linesplit[7])],
                [float(linesplit[8]),float(linesplit[9]),float(linesplit[10]),float(linesplit[11])]])
        calibs.append(calib)
    return calibs

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
    velo2cam = np.array([[7.533745000000e-03,-9.999714000000e-01,-6.166020000000e-04,-4.069766000000e-03],
                        [1.480249000000e-02,7.280733000000e-04,-9.998902000000e-01,-7.631618000000e-02],
                        [9.998621000000e-01,7.523790000000e-03,1.480755000000e-02,-2.717806000000e-01],
                        [0,0,0,1]])          
    R0_rect = np.array([[9.999239000000e-01,9.837760000000e-03,-7.445048000000e-03,0],
                        [-9.869795000000e-03,9.999421000000e-01,-4.278459000000e-03,0],
                        [7.402527000000e-03,4.351614000000e-03,9.999631000000e-01,0],
                        [0,0,0,1]])
    sem_pallete = yaml.safe_load(open('ws_hiwi/semantic-kitti.yaml', 'r'))["color_map"]
    path = args.seqpath
    path_poses = path + 'poses.txt'
    path_calib = path + 'calib.txt'
    poses = read_poses(path_poses)
    p2 = read_calib(path_calib)[2]
    sem_colors = np.array([[0,0,0]])
    sem_scans = np.array([[0,0,0]])
    volume_colors = np.array([[0,0,0,0,0,0]])
    rgb_scans = np.array([[0,0,0,0,0,0]])
    
    startframe = args.startframe
    endframe = args.endframe
    for i in tqdm(range(startframe, endframe)):
        path_pts = path + 'velodyne/' + idx2str(i) +'.bin'
        path_labels = path + 'labels/' + idx2str(i) +'.label'
        path_imgs = path + 'image_2/' + idx2str(i) +'.png'    

        # point cloud
        currentscan = read_bin(path_pts)
        pose_velo = np.linalg.inv(velo2cam).dot(poses[i].dot(velo2cam))
        currentscan_gloabl = dot(pose_velo, currentscan)[:,0:3]

        # color
        img = image.imread(path_imgs)
        img_coords_homo = dot(p2.dot(R0_rect).dot(velo2cam),currentscan)
        currentcolor = np.zeros([len(currentscan),3])
        for idx_img, img_coord in enumerate(img_coords_homo):
            if img_coord[1]/img_coord[2] > 0 and img_coord[1]/img_coord[2] < img.shape[0] and img_coord[0]/img_coord[2] > 0 and img_coord[0]/img_coord[2] < img.shape[1] and img_coord[2] > 0:
                currentcolor[idx_img] = img[int(img_coord[1]/img_coord[2])][int(img_coord[0]/img_coord[2])]
            else:
                currentcolor[idx_img] = np.array([[0,0,0]])
        rgb_scan = np.concatenate([currentscan_gloabl, currentcolor],1)
        rgb_scan_mask = (rgb_scan[:, 3] > 0)
        rgb_scan_masked =rgb_scan[rgb_scan_mask, :]
        if not args.iffusion:  
            dvis(rgb_scan_masked, l=1,t=i,vs=0.02, name='pts/rgb scan'+str(i))

        # semantic pallet
        currentlabel = read_label(path_labels)
        currentsemcolor = np.zeros([len(currentscan),3])
        for idx, pointlabel in enumerate(currentlabel):
            semantic_pointlabel = pointlabel & 0xFF #take out lower 16 bits
            if semantic_pointlabel in sem_pallete:
                currentsemcolor[idx] = np.asarray(sem_pallete[semantic_pointlabel])
            else:
                currentsemcolor[idx] = np.array([0,0,0])
        if not args.iffusion:  
            dvis(np.concatenate([currentscan_gloabl, currentsemcolor],1),l=2,t=i,vs=0.10, name='pts/semantic scan'+str(i))
        

        # visualize groundtruth
        path_volume = path + 'voxels/' + idx2str(i) +'.occluded'
        path_volume_labels = path + 'voxels/' + idx2str(i) +'.label'
        volume = unpack(np.fromfile(path_volume, dtype=np.uint8)).reshape(256,256,32)
        labels = np.fromfile(path_volume_labels, dtype=np.uint16).reshape(256,256,32)
        
        valid_label_mask = labels>0
        valid_labels = labels[valid_label_mask]
        valid_label_inds = np.stack(np.nonzero(valid_label_mask),1)

        vox2scene = np.eye(4)
        vox2scene[:3,:3] = np.diag([1/5,1/5,1/5])
        vox2scene[:3,3] = np.array([0.1,-25.5,-1.9])
        valid_scene_coords = dot(vox2scene, valid_label_inds)
        valid_scene_coords_global = valid_scene_coords.dot(pose_velo.T)[:,0:3]

        valid_colors = np.zeros((len(valid_scene_coords),3))
        for label in np.unique(valid_labels):
            if label>0:
                label_mask = valid_labels == label
                valid_colors[label_mask] = sem_pallete[label]

        valid_scene_coords_col = np.concatenate([valid_scene_coords_global,valid_colors],1)
        if not args.iffusion:  
            dvis(valid_scene_coords_col, l=3, t=i, vs=1/5, name='volume/semantic volume'+ str(i))

        if args.iffusion:  
            sem_colors = np.concatenate([sem_colors, currentsemcolor],0)
            sem_scans = np.concatenate([sem_scans, currentscan_gloabl],0)
            volume_colors = np.concatenate([volume_colors, valid_scene_coords_col],0)
            rgb_scans = np.concatenate([rgb_scans, rgb_scan_masked],0)

    if args.iffusion:
        dvis(rgb_scans, l=1, vs=0.03, ms=1000000, name='pts/fused rgb pts')
        dvis(np.concatenate([sem_scans, sem_colors],1),l=2,vs=0.10, ms=1000000,name='pts/fused semantic pts')
        dvis(volume_colors, l=3, vs=1/5, ms=1000000, name='volume/fused volume')
        
    

if __name__=="__main__":
    main()