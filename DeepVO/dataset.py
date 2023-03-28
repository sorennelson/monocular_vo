import random
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from streaming import StreamingDataset
from typing import Callable, Any

def get_training_data_loaders(remote_path, local_path, batch_size, n_src, n_workers):
    train_transforms = None #transforms
    test_transforms = None

    train_data = KittiDataset(f'{remote_path}/train', f'{local_path}/train', 
                              shuffle=True, 
                              batch_size=batch_size, 
                              n_src=n_src,
                              transforms=train_transforms)
    test_data = KittiDataset(f'{remote_path}/val', f'{local_path}/val', 
                             shuffle=False, 
                             batch_size=batch_size, 
                             n_src=n_src,
                             transforms=test_transforms)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, 
                              num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             num_workers=n_workers, pin_memory=True)
    
    return train_loader, test_loader


def get_test_data_loader(remote_path, local_path, batch_size, n_src, n_workers):
    test_data = KittiDataset(f'{remote_path}/test', f'{local_path}/test', 
                              shuffle=False, 
                              batch_size=batch_size, 
                              n_src=n_src,
                              transforms=None,
                              src1_origin=True)
    
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             num_workers=n_workers, pin_memory=True)
    
    return test_loader


# def transforms_h(target, src, cam):
#     def horizontal_flip(imgs, cam):
#         if random.random() < 0.5:
#             imgs = TF.hflip(imgs)
#             cam[0,2] = imgs.shape[2] - cam[0,2]
#         return imgs, cam

#     imgs = torch.cat([target, src], dim=0)
#     imgs, cam = horizontal_flip(imgs, cam)


# def transforms(target, src, cam):

#     def crop_resize(imgs, cam):
#         h, w = imgs.shape[1], imgs.shape[2]
#         # Resize
#         r_scale_y, r_scale_x = random.uniform(1., 1.15), random.uniform(1., 1.15) 
#         res_h, res_w = int(h*r_scale_y), int(w*r_scale_x)
#         imgs = TF.resize(imgs, (res_h, res_w))
#         # Crop
#         crop_top, crop_left = int(random.uniform(0, res_h-h+1)), int(random.uniform(0, res_w-w+1))
#         imgs = TF.crop(imgs, crop_top, crop_left, h, w)
#         # Adjust intrinsics
#         cam[0,0] *= r_scale_x
#         cam[1,1] *= r_scale_y
#         cam[0,2] = cam[0,2] * r_scale_x - crop_left
#         cam[1,2] = cam[1,2] * r_scale_y - crop_top
#         return imgs, cam
    
#     imgs = torch.cat([target, src], dim=0)
#     imgs, cam = crop_resize(imgs, cam)

#     return imgs[:1], imgs[1:], cam


class KittiDataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable,
                 n_src: int = 4,
                 src1_origin: bool = False
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
        target = torch.unsqueeze(seq[:,t_start:t_end],0) # Middle image
        # TODO: Verify
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

        n_pre = self.n_src//2
        poses = torch.zeros(self.n_src,3,4)
        # pose_pre
        for i in range(n_pre):
            # Unpack global pose (poses are stored as [seq, idx, pose])
            pose_src = np.array(pose_seq[i].split(',')[2:]).reshape(3,4)
            pose_src = torch.tensor(pose_src.astype(np.float32))
            poses[i] = pose_src
        # pose_post
        for i in range(n_pre+1,self.n_src+1):
            # Unpack global pose (poses are stored as [seq, idx, pose])
            pose_src = np.array(pose_seq[i].split(',')[2:]).reshape(3,4)
            pose_src = torch.tensor(pose_src.astype(np.float32))
            poses[i-1] = pose_src

        pose_target = np.array(pose_seq[n_pre].split(',')[2:]).reshape(3,4)
        pose_target = torch.tensor(pose_target.astype(np.float32))
        R_target, t_target = pose_target[:,:-1], pose_target[:,-1:]
        
        R_src, t_src = poses[:,:,:-1], poses[:,:,-1:]
        poses = torch.cat([
            R_src.transpose(1,2) @ R_target, R_src.transpose(1,2) @ (t_target - t_src)
            ], dim=2)

        return poses
    
    def _extract_src1_orgin_pose_from_sequence(self, pose_seq):
        pose_seq = pose_seq.split('|')

        poses = torch.zeros(self.n_src+1,3,4)
        # pose_pre
        for i in range(self.n_src+1):
            # Unpack global pose (poses are stored as [seq, idx, pose])
            pose = np.array(pose_seq[i].split(',')[2:]).reshape(3,4)
            pose = torch.tensor(pose.astype(np.float32))
            poses[i] = pose

        # Src_1 as origin
        R_src1, t_src1 = poses[0,:,:-1], poses[0,:,-1]
        poses[:,:,-1] -= t_src1
        poses = torch.linalg.inv(R_src1) @ poses

        return poses