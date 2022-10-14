## Read the visualize the data from the file
from dvis import dvis
import numpy as np
import yaml
import argparse

#SEQ = ['08','11','12','13','14','15','16','17','18','19','20','21']

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--seq', type=str, default='08', help='choose the sequence to visualize')
    parser.add_argument('--predall', type=bool, default=False, help='if visualize all the catagories')
    parser.add_argument('--predstatic', type=bool, default=False, help='if visualize the static catagories')
    parser.add_argument('--input', type=bool, default=False, help='if visualize the input')
    parser.add_argument('--pose', type=bool, default=False, help='if load the pose')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.predall:
        color_dict = yaml.safe_load(open('ws_hiwi/semantic-kitti.yaml', 'r'))["color_map"]
        path_pred = 'ws_hiwi/seq'+ args.seq +'_static.npy'
        volume_pred = np.load(path_pred)
        valid_colors = np.zeros((len(volume_pred),3))
        for label in np.unique(volume_pred[:,3]):
            label_mask = volume_pred[:,3] == label
            valid_colors[label_mask] = color_dict[label]
        volume_colors = np.concatenate([volume_pred[:,:3],valid_colors],1)
        dvis(volume_colors, l=1, vs=1/5, ms=1000000, name='volume/pred')
    if args.predstatic:
        color_dict = yaml.safe_load(open('ws_hiwi/semantic-kitti.yaml', 'r'))["color_map"]
        path_pred = 'ws_hiwi/seq'+ args.seq +'_alllabel.npy'
        volume_pred = np.load(path_pred)
        valid_colors = np.zeros((len(volume_pred),3))
        for label in np.unique(volume_pred[:,3]):
            label_mask = volume_pred[:,3] == label
            valid_colors[label_mask] = color_dict[label]
        volume_colors = np.concatenate([volume_pred[:,:3],valid_colors],1)
        dvis(volume_colors, l=2, vs=1/5, ms=1000000, name='volume/pred')
    if args.input:
        path_in = 'ws_hiwi/seq'+ args.seq +'_input.npy'
        volume_input = np.load(path_in)
        dvis(volume_input[:,1:], l=3, vs=1/5, ms=1000000, name='volume/input')
    if args.pose:
        path_poses = 'ws_hiwi/seq'+ args.seq +'_pose.npy'
        poses = np.load(path_poses)
    

if __name__=="__main__":
    main()