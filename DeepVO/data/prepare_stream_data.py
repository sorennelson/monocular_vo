import numpy as np
import cv2
from PIL import Image
from streaming import MDSWriter
import os

# Local or remote directory in which to store the compressed output files
train_dir = '../kitti_vo_streaming_tmp/train'
val_dir = '../kitti_vo_streaming_tmp/val'
in_dir = '../kitti_vo'

# A dictionary mapping input fields to their data types
columns = {
    'image': 'jpeg',
    'cam': 'str'
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
    all_list = {}
    all_list['image_file_list'] = image_file_list
    all_list['cam_file_list'] = cam_file_list
    return all_list

def write_dataset(dataset, out_dir) -> None:
    # Save the samples as shards using MDSWriter
    with MDSWriter(out=out_dir, columns=columns, compression=compression, keep_local=True) as out:
        for i in range(len(dataset['image_file_list'])):
            print(Image.fromarray(cv2.imread(dataset['image_file_list'][i], 0)))
            return
        #     with open(dataset['cam_file_list'][i], 'r') as f:
        #         cam = f.readline()
        #         assert len(cam.split(',')) == 9, f'{i}: {len(cam.split(",")) }'
        #     sample = {
        #         'image': Image.fromarray(cv2.imread(dataset['image_file_list'][i], 0)),
        #         'cam': cam,
        #     }
        #     out.write(sample)

# train_ds = format_file_list(in_dir, 'train')
val_ds = format_file_list(in_dir, 'val')

# write_dataset(train_ds, train_dir)
write_dataset(val_ds, val_dir)