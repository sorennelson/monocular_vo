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
    # test_data = KittiDataset(f'{remote_path}/val', f'{local_path}/val', 
    #                          shuffle=False, 
    #                          batch_size=batch_size, 
    #                          transforms=test_transforms)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, 
                              num_workers=n_workers, pin_memory=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size,
    #                          num_workers=n_workers, pin_memory=True)
    
    return train_loader, None #test_loader

class KittiDataset(StreamingDataset):
    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 batch_size: int,
                 transforms: Callable
                ) -> None:
        super().__init__(local=local, remote=remote, 
                         shuffle=shuffle, 
                         batch_size=batch_size)
        self.transforms = transforms

    def __getitem__(self, idx:int) -> Any:
        obj = super().__getitem__(idx)
        img_sequence = torch.tensor(np.array(obj['image']))
        target, src = self._unpack_img_sequence(img_sequence)
        target = self._preprocess_img(target)
        src = self._preprocess_img(src)

        cam = np.array(obj['cam'].split(',')).reshape(3,3)
        cam = torch.tensor(cam.astype(np.float32))
        
        # return self.transforms(x), y
        return target, src, cam
    
    def _unpack_img_sequence(self, seq, width=416):
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