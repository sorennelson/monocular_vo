from __future__ import division
import numpy as np
from glob import glob
import os
import cv2

class kitti_odom_loader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=5):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.test_seqs = [9, 10]

        self.collect_test_frames()
        self.collect_train_frames()

    def collect_test_frames(self):
        self.test_frames = []
        for seq in self.test_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_0')
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.test_frames.append('%.2d %.6d' % (seq, n))
        self.num_test = len(self.test_frames)
        self.collect_test_poses()
        assert len(self.test_poses) == self.num_test

    def collect_test_poses(self):
        self.test_poses = []
        for seq in self.test_seqs:
            with open(os.path.join(self.dataset_dir, 'poses', '%.2d.txt' % seq), 'r') as f:
                seq_poses = [line.strip().split() for line in f.readlines()]
                for i in range(len(seq_poses)):
                    seq_poses[i].insert(0, i)
                    seq_poses[i].insert(0, seq)
                self.test_poses.extend(seq_poses)
        
    def collect_train_frames(self):
        self.train_frames = []
        for seq in self.train_seqs:
            seq_dir = os.path.join(self.dataset_dir, 'sequences', '%.2d' % seq)
            img_dir = os.path.join(seq_dir, 'image_0')
            N = len(glob(img_dir + '/*.png'))
            for n in range(N):
                self.train_frames.append('%.2d %.6d' % (seq, n))
        self.num_train = len(self.train_frames)
        self.collect_train_poses()
        assert len(self.train_poses) == self.num_train

    def collect_train_poses(self):
        self.train_poses = []
        for seq in self.train_seqs:
            with open(os.path.join(self.dataset_dir, 'poses', '%.2d.txt' % seq), 'r') as f:
                seq_poses = [line.strip().split() for line in f.readlines()]
                for i in range(len(seq_poses)):
                    seq_poses[i].insert(0, i)
                    seq_poses[i].insert(0, seq)
                self.train_poses.extend(seq_poses)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, _ = frames[min_src_idx].split(' ')
        max_src_drive, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
            return True
        return False

    def load_image_sequence(self, frames, poses, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        pose_seq = []
        for o in range(-half_offset, half_offset+1):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image(curr_drive, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = cv2.resize(curr_img, (self.img_width,self.img_height))
            image_seq.append(curr_img)
            pose_seq.append(poses[curr_idx])
        return image_seq, zoom_x, zoom_y, pose_seq

    def load_example(self, frames, poses, tgt_idx, load_pose=False):
        image_seq, zoom_x, zoom_y, pose_seq = self.load_image_sequence(
            frames, poses, tgt_idx, self.seq_length
            )
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')

        intrinsics = self.load_intrinsics(tgt_drive, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)        
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        example['pose_seq'] = pose_seq
        if load_pose:
            pass
        return example

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, self.train_poses, tgt_idx)
        return example
    
    def get_test_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.test_frames, tgt_idx):
            return False
        example = self.load_example(self.test_frames, self.test_poses, tgt_idx)
        return example

    def load_image(self, drive, frame_id):
        img_file = os.path.join(self.dataset_dir, 'sequences', '%s/image_0/%s.png' % (drive, frame_id))
        img = cv2.imread(img_file, 0)
        return img

    def load_intrinsics(self, drive, frame_id):
        calib_file = os.path.join(self.dataset_dir, 'sequences', '%s/calib.txt' % drive)
        proj_c2p, _ = self.read_calib_file(calib_file)
        intrinsics = proj_c2p[:3, :3]
        return intrinsics

    def read_calib_file(self, filepath, cid=0):
        """Read in a calibration file and parse into a dictionary."""
        with open(filepath, 'r') as f:
            C = f.readlines()
        def parseLine(L, shape):
            data = L.split()
            data = np.array(data[1:]).reshape(shape).astype(np.float32)
            return data
        proj_c2p = parseLine(C[cid], shape=(3,4))
        proj_v2c = parseLine(C[-1], shape=(3,4))
        filler = np.array([0, 0, 0, 1]).reshape((1,4))
        proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
        return proj_c2p, proj_v2c

    def scale_intrinsics(self,mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out


