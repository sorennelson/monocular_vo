import numpy as np
import cv2
from PIL import Image
from streaming import MDSWriter
import os, argparse

# Local or remote directory in which to store the compressed output files
# train_dir = '../kitti_vo_streaming_pose_4src/train'
# val_dir = '../kitti_vo_streaming_pose_4src/val'
# test_dir = '../kitti_vo_streaming_pose_4src/test'
# in_dir = '../kitti_vo_pose_4src'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, 
                    help="Where the dataset is stored (dump_root from prepare_[split]_data.py)")
parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
parser.add_argument("--dump_root", type=str, required=True, 
                    help="Local or remote directory in which to store the compressed output files.")
args = parser.parse_args()


# A dictionary mapping input fields to their data types
columns = {
    'image': 'jpeg',
    'cam': 'str',
    'pose': 'str'
}

# Shard compression, if any
compression = 'zstd'


def format_file_list(data_root, split):
    with open(data_root + '/%s.txt' % split, 'r') as f:
        frames = f.readlines()
    subfolders = [x.split(' ')[0] for x in frames]
    frame_ids = [x.split(' ')[1][:-1] for x in frames]
    image_file_list = [os.path.join(data_root, subfolders[i], 
        frame_ids[i] + '.jpg') for i in range(len(frames))]
    cam_file_list = [os.path.join(data_root, subfolders[i], 
        frame_ids[i] + '_cam.txt') for i in range(len(frames))]
    pose_file_list = [os.path.join(data_root, subfolders[i], 
        frame_ids[i] + '_pose.txt') for i in range(len(frames))]
    all_list = {}
    all_list['image_file_list'] = image_file_list
    all_list['cam_file_list'] = cam_file_list
    all_list['pose_file_list'] = pose_file_list
    return all_list

def write_dataset(dataset, out_dir) -> None:
    # Save the samples as shards using MDSWriter
    with MDSWriter(out=out_dir, columns=columns, compression=compression, keep_local=True) as out:
        for i in range(len(dataset['image_file_list'])):
            with open(dataset['cam_file_list'][i], 'r') as f:
                cam = f.readline()
                assert len(cam.split(',')) == 9, f'{i}: {len(cam.split(",")) }'
            with open(dataset['pose_file_list'][i], 'r') as f:
                pose = f.readline()
            sample = {
                'image': Image.fromarray(cv2.imread(dataset['image_file_list'][i], 0)),
                'cam': cam,
                'pose': pose
            }
            out.write(sample)

ds = format_file_list(args.dataset_dir, args.split)
write_dataset(ds, os.path.join(args.dump_root, args.split))

# # train_ds = format_file_list(in_dir, 'train')
# val_ds = format_file_list(in_dir, 'val')
# # test_ds = format_file_list(in_dir, 'test')

# # write_dataset(train_ds, train_dir)
# write_dataset(val_ds, val_dir)
# # write_dataset(test_ds, test_dir)