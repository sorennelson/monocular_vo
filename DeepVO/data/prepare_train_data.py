from __future__ import division
import argparse
import scipy.misc
import cv2
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, required=True, choices=["kitti_odom"])
parser.add_argument("--dump_root", type=str, required=True, help="Where to dump the data")
parser.add_argument("--seq_length", type=int, required=True, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        im = np.expand_dims(im, -1)
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def dump_example(n, args):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)

    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(args.dump_root, example['folder_name'])
    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    cv2.imwrite(dump_img_file, image_seq.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))
    dump_pose_file = dump_dir + '/%s_pose.txt' % example['file_name']
    with open(dump_pose_file, 'w') as f:
        if args.seq_length == 5:
            src1_pose = ','.join([str(x) for x in example['pose_seq'][0]])
            src2_pose = ','.join([str(x) for x in example['pose_seq'][1]])
            targ_pose = ','.join([str(x) for x in example['pose_seq'][2]])
            src3_pose = ','.join([str(x) for x in example['pose_seq'][3]])
            src4_pose = ','.join([str(x) for x in example['pose_seq'][4]])
            f.write(f'{src1_pose}|{src2_pose}|{targ_pose}|{src3_pose}|{src4_pose}')
        elif args.seq_length == 3:
            src1_pose = ','.join([str(x) for x in example['pose_seq'][0]])
            targ_pose = ','.join([str(x) for x in example['pose_seq'][1]])
            src2_pose = ','.join([str(x) for x in example['pose_seq'][2]])
            f.write(f'{src1_pose}|{targ_pose}|{src2_pose}')

def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader
    if args.dataset_name == 'kitti_odom':
        from kitti.kitti_odom_loader import kitti_odom_loader
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))

    # Split into train/val
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    with open(args.dump_root + 'train.txt', 'w') as tf:
        with open(args.dump_root + 'val.txt', 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))

main()

