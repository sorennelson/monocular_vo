import torch
import numpy as np
from torch.utils.data import DataLoader
from streaming import StreamingDataset
from typing import Callable, Any

def get_training_data_loaders(remote_path, 
                              local_path, 
                              train_batch_size, 
                              n_src, 
                              n_workers):
    train_transforms = None
    test_transforms = None

    train_data = KittiDataset(f'{remote_path}/train', f'{local_path}/train', 
                              shuffle=True, 
                              batch_size=train_batch_size, 
                              n_src=n_src,
                              transforms=train_transforms,
                              src1_origin=True)
    test_data = KittiDataset(f'{remote_path}/val', f'{local_path}/val', 
                             shuffle=False, 
                             batch_size=1, 
                             n_src=n_src,
                             transforms=test_transforms,
                             src1_origin=True)
    
    train_loader = DataLoader(train_data, batch_size=train_batch_size, 
                              num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=1,
                             num_workers=n_workers, pin_memory=True)
    
    return train_loader, test_loader


def get_test_data_loader(remote_path, local_path, n_src, n_workers):
    test_data = KittiDataset(f'{remote_path}/test', f'{local_path}/test', 
                              shuffle=False, 
                              batch_size=1, 
                              n_src=n_src,
                              transforms=None,
                              src1_origin=True)
    
    test_loader = DataLoader(test_data, batch_size=1,
                             num_workers=n_workers, pin_memory=True)
    
    return test_loader


class KittiDataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable,
                 n_src: int = 4,
                 src1_origin: bool = True
                ) -> None:
        super().__init__(local=local, remote=remote, 
                         shuffle=shuffle, 
                         batch_size=batch_size)
        self.transforms = transforms
        assert n_src % 2 == 0, 'Must have even number of source images'
        self.n_src = n_src
        self.src1_origin = src1_origin

    def __getitem__(self, idx:int) -> Any:
        obj = super().__getitem__(idx)
        img_sequence = torch.tensor(np.array(obj['image']))
        target, src = self._unpack_img_sequence(img_sequence)
        target = self._preprocess_img(target)
        src = self._preprocess_img(src)

        cam = np.array(obj['cam'].split(',')).reshape(3,3)
        cam = torch.tensor(cam.astype(np.float32))
        cam = self._get_multi_scale_intrinsics(cam, 4)

        if not self.src1_origin:
            pose = self._extract_target_origin_pose_from_sequence(obj['pose'])
        else:
            pose = self._extract_src1_orgin_pose_from_sequence(obj['pose'])
        
        if self.transforms:
            target, src, cam = self.transforms(target, src, cam)
        
        return target, src, cam, pose
    
    def _unpack_img_sequence(self, seq:torch.Tensor, width=416):
        '''Unpacks images stacked horizontally as [src_pre,target,src_post]. 
        Returns target, [src_pre,src_post] (concatenated along channel dimension).
        '''
        n_pre = self.n_src//2
        t_start, t_end = width*n_pre, width*(n_pre+1)
        # Middle image
        target = torch.unsqueeze(seq[:,t_start:t_end],0)
        src_imgs = []
        # src_pre
        for i in range(n_pre):
            src_imgs.append(seq[:,width*i:width*(i+1)])
        # src_post
        for i in range(n_pre+1,self.n_src+1):
            src_imgs.append(seq[:,width*i:width*(i+1)]) 
        src = torch.stack(src_imgs, dim=0)
        return target, src
    
    def _preprocess_img(self, img):
        '''Convert to range [-1,1]'''
        return img.type(torch.float32) / 128. - 1.
    
    def _deprocess_img(self, img):
        '''Convert to range [0,256]'''
        return ((img+1)*128).type(torch.int8)

    def _get_multi_scale_intrinsics(self, 
                                    intrinsics: torch.Tensor, 
                                    n_scales: int):
        ''' 
        Return n_scale intrinsics Tensor with adjusted parameters.
        Args:
            intrinsics: camera intrinsics at the normal scale (3,3)
            n_scales: number of scales
        Returns:
            torch.Tensor (n_scales, 3, 3)
        '''
        intrinsics = intrinsics.unsqueeze(0)
        multi_scale_intrinsics = intrinsics.repeat(n_scales,1,1)
        for s in range(1,n_scales):
            multi_scale_intrinsics[s,:-1] /= (2**s)
        return multi_scale_intrinsics


    def _extract_target_origin_pose_from_sequence(self, pose_seq):
        pose_seq = pose_seq.split('|')

        poses = torch.zeros(self.n_src+1,4,4)
        for i in range(self.n_src+1):
            # Unpack global pose (poses are stored as [seq, idx, pose])
            pose = np.array(pose_seq[i].split(',')[2:]).reshape(3,4)
            pose = torch.tensor(pose.astype(np.float32))
            poses[i] = torch.cat([pose, torch.tensor([[0,0,0,1]])], dim=0)

        n_pre = self.n_src // 2
        target_orgin_poses = torch.zeros(self.n_src,3,4)
        target_pose_inv = torch.linalg.inv(poses[n_pre])
        for i, pose in enumerate(poses):
            if i == n_pre: continue
            j = i if i < n_pre else i-1
            target_orgin_poses[j] = (target_pose_inv @ pose)[:-1]

    
    def _extract_src1_orgin_pose_from_sequence(self, pose_seq):
        pose_seq = pose_seq.split('|')

        poses = torch.zeros(self.n_src+1,4,4)
        for i in range(self.n_src+1):
            # Unpack global pose (poses are stored as [seq, idx, pose])
            pose = np.array(pose_seq[i].split(',')[2:]).reshape(3,4)
            pose = torch.tensor(pose.astype(np.float32))
            poses[i] = torch.cat([pose, torch.tensor([[0,0,0,1]])], dim=0)
        
        # Convert to src1 origin
        src1_origin_poses = (torch.linalg.inv(poses[0]) @ poses)[:,:-1]
        return src1_origin_poses