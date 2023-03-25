import torch
import numpy as np
from torch.utils.data import DataLoader
from streaming import StreamingDataset
from typing import Callable, Any

def get_training_data_loaders(remote_path, local_path, batch_size, n_workers):
    # TODO
    train_transforms = []
    test_transforms = []

    train_data = KittiDataset(f'{remote_path}/train', f'{local_path}/train', 
                              shuffle=True, 
                              batch_size=batch_size, 
                              transforms=train_transforms)
    test_data = KittiDataset(f'{remote_path}/val', f'{local_path}/val', 
                             shuffle=False, 
                             batch_size=batch_size, 
                             transforms=test_transforms)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, 
                              num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             num_workers=n_workers, pin_memory=True)
    
    return train_loader, test_loader

def get_test_data_loader(remote_path, local_path, batch_size, n_workers):
    test_data = KittiDataset(f'{remote_path}/test', f'{local_path}/test', 
                              shuffle=False, 
                              batch_size=batch_size, 
                              transforms=[],
                              global_pose=True)
    
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             num_workers=n_workers, pin_memory=True)
    
    return test_loader


class KittiDataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable,
                 global_pose: bool = False
                ) -> None:
        super().__init__(local=local, remote=remote, 
                         shuffle=shuffle, 
                         batch_size=batch_size)
        self.transforms = transforms
        self.global_pose = global_pose

    def __getitem__(self, idx:int) -> Any:
        obj = super().__getitem__(idx)
        img_sequence = torch.tensor(np.array(obj['image']))
        target, src = self._unpack_img_sequence(img_sequence)
        target = self._preprocess_img(target)
        src = self._preprocess_img(src)

        cam = np.array(obj['cam'].split(',')).reshape(3,3)
        cam = torch.tensor(cam.astype(np.float32))

        if not self.global_pose:
            pose = self._extract_relative_pose_from_sequence(obj['pose'])
        else:
            pose = self._extract_global_pose_from_sequence(obj['pose'])
        
        # return self.transforms(x), y
        return target, src, cam, pose
    
    def _unpack_img_sequence(self, seq:torch.Tensor, width=416):
        '''Unpacks images stacked horizontally as [src1,target,src2]. 
        Returns target, [src1,src2] (concatenated along channel dimension).
        '''
        t_start, t_end = width, width*2
        target = torch.unsqueeze(seq[:,t_start:t_end],0) # Middle image
        src = torch.stack([seq[:,:t_start], seq[:,t_end:]], dim=0)
        return target, src
    
    def _preprocess_img(self, img):
        '''Convert to range [-1,1]'''
        return img.type(torch.float32) / 128. - 1.
    
    def _deprocess_img(self, img):
        '''Convert to range [0,256]'''
        return ((img+1)*128).type(torch.int8)

    def _augment_imgs(self, target, src, cam):
        pass

    def _get_multi_scale_intrinsics(self, intrinsics, num_scales):
        pass

    def _extract_relative_pose_from_sequence(self, pose_seq):
        pose_seq = pose_seq.split('|')
        # Unpack global pose (poses are stored as [seq, idx, pose])
        pose_src1 = np.array(pose_seq[0].split(',')[2:]).reshape(3,4)
        pose_src1 = torch.tensor(pose_src1.astype(np.float32))
        R_src1, t_src1 = pose_src1[:,:-1], pose_src1[:,-1:]

        pose_target = np.array(pose_seq[1].split(',')[2:]).reshape(3,4)
        pose_target = torch.tensor(pose_target.astype(np.float32))
        R_target, t_target = pose_target[:,:-1], pose_target[:,-1:]

        pose_src2 = np.array(pose_seq[2].split(',')[2:]).reshape(3,4)
        pose_src2 = torch.tensor(pose_src2.astype(np.float32))
        R_src2, t_src2 = pose_src2[:,:-1], pose_src2[:,-1:]
        
        # Convert to relative pose w.r.t target
        #   R_target^T @ R_src | R_target^T @ (t_src - t_target)
        pose_src1_rel = torch.cat([
            R_src1.T @ R_target, R_src1.T @ (t_target - t_src1)
            ], dim=1)
        pose_src2_rel = torch.cat([
            R_src2.T @ R_target, R_src2.T @ (t_target - t_src2)
            ], dim=1)
        
        return torch.stack([
            pose_src1_rel, pose_src2_rel
        ], dim=0)

    def _extract_global_pose_from_sequence(self, pose_seq):
        pose_seq = pose_seq.split('|')
        # Unpack global pose and seq idx (poses are stored as [seq, idx, pose])
        pose_src1 = np.array(pose_seq[0].split(',')[2:]).reshape(3,4)
        pose_src1 = torch.tensor(pose_src1.astype(np.float32))
        idx_src1 = np.array(pose_seq[0].split(',')[:2])
        idx_src1 = torch.tensor(idx_src1.astype(np.int32), dtype=torch.int)

        pose_target = np.array(pose_seq[1].split(',')[2:]).reshape(3,4)
        pose_target = torch.tensor(pose_target.astype(np.float32))
        idx_target = np.array(pose_seq[1].split(',')[:2])
        idx_target = torch.tensor(idx_target.astype(np.int32), dtype=torch.int)

        pose_src2 = np.array(pose_seq[2].split(',')[2:]).reshape(3,4)
        pose_src2 = torch.tensor(pose_src2.astype(np.float32))
        idx_src2 = np.array(pose_seq[2].split(',')[:2])
        idx_src2 = torch.tensor(idx_src2.astype(np.int32), dtype=torch.int)

        poses = torch.stack([
            pose_src1, pose_target, pose_src2
        ], dim=0)
        
        indexing = torch.stack([
            idx_src1, idx_target, idx_src2
        ], dim=0)

        return poses, indexing