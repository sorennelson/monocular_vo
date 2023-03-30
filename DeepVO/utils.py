import os, time
import torch
import torch.nn as nn


class MetricLogger:
    ''' Helper class for storing metrics and logging. '''

    def __init__(self, save_path:str, job_id:str, batch_size:int, epochs:int):
        self.reset_metrics()
        self.epoch = 0
        self.start = None
        self.batch_size = batch_size
        self.epochs = epochs
        if not os.path.isdir(os.path.join(save_path, job_id)):
            os.makedirs(os.path.join(save_path, job_id))
        self.log = open(os.path.join(save_path, job_id, 'log.txt'), 'w')
        
    def start_epoch(self, epoch:int):
        self.start = time.time()
        self.epoch = epoch
        self.reset_metrics()
        self.log.write(f'Training epoch {epoch}/{self.epochs} ...\n')
        self.log.flush()

    def reset_metrics(self):
        self.loss = 0.
        self.pose_dist = 0.
        self.n_examples = 0

    def update(self, batch_size:int, loss:float, pose_dist:float):
        ''' Update with iteration stats. '''
        self.loss += loss*batch_size
        self.pose_dist += pose_dist*batch_size
        self.n_examples += batch_size

    def log_str(self, s: str):
        self.log.write(f'{s}\n')
        self.log.flush()

    def log_stats(self):
        self.log.write(
            f'..... Epoch {self.epoch}/{self.epochs}, ' +
            f'iter: {self.n_examples // self.batch_size}, ' +
            f'time: {int((time.time() - self.start) // 60)}, ' +
            f'loss: {self.loss/self.n_examples:.3f}, ' +
            f'ATE: {self.pose_dist/self.n_examples:.3f}\n')
        self.log.flush()
    
    def log_epoch_stats(self, train:bool):
        if train:
            epoch_time = time.time() - self.start
            rem_time = epoch_time*(self.epochs-self.epoch)
            rem_minutes = (rem_time // 60) % 60
            rem_hours = rem_time // 60 // 60  
            self.log.write(
                f'==> Epoch {self.epoch}/{self.epochs}, ' +
                f'iters: {self.n_examples // self.batch_size}, ' +
                f'time: {int(epoch_time // 60)}, ' +
                f'est time remaining: {int(rem_hours)}H {int(rem_minutes)}M  --- ' +
                f'avg train loss: {self.loss/self.n_examples:.3f}, ' +
                f'avg train ATE: {self.pose_dist/self.n_examples:.3f}\n')
        else:
            self.log.write(
                f'==> Validation --- ' +
                f'avg val loss: {self.loss/self.n_examples:.3f}, ' +
                f'avg val ATE: {self.pose_dist/self.n_examples:.3f}\n\n')
        self.log.flush()


# Loss Helpers

def get_zeros(shape, device):
    return torch.zeros(shape, device=device)

def get_ones(shape, device):
    return torch.ones(shape, device=device)

def get_pose_pad(batch_size, device):
    x = get_zeros((batch_size,1,4), device)
    x[:,0,-1] = 1
    return x


def projective_inverse_warp(src:torch.Tensor, 
                            depth:torch.Tensor, 
                            pose:torch.Tensor, 
                            cam:torch.Tensor):
    ''' 
    Projects source onto target view using predicted depth and camera pose 
    then warps the projected pixel values using bilinear interpolation.

    Args:
        src: Batch images for single source view (B,1,H,W)
        depth: Depth map prediction (B,1,H,W)
        pose: 6DOF pose predictions as target->src (B,n_src,6)
        cam: Camera intrinsics (B,3,3)
    '''
    # P = K @ [R|t]_t->s (Camera Projection Matrix)
    pose_mat = get_pose_mat(pose)
    P = torch.matmul(cam, pose_mat)
    
    # X = D @ K^-1 @ p_t (3D point)
    # p_t - homogeneous coords of pixels in target view 
    #   (just passing in src here for convenience since src.shape == target.shape)
    p_t = get_p_t(src)
    cam_coords = torch.linalg.solve(cam, p_t.view(src.shape[0],3,-1))
    cam_coords = cam_coords.view(src.shape[0],3,*src.shape[2:])
    X = cam_coords * depth
    # Homogenous transform
    X = torch.cat([
        X, get_ones((X.shape[0], 1, *X.shape[2:]), X.device)
        ], dim=1)
    
    # p_s = P @ X (Projected coords of pixels in source view)
    p_s = torch.matmul(P, X.view(src.shape[0], 4, -1))
    # Non-homogenous transform
    p_s = torch.cat([
        p_s[:,:1] / (p_s[:,2:3] + 1e-10),
        p_s[:,1:2] / (p_s[:,2:3] + 1e-10)
    ], axis=1).view(src.shape[0],2,*src.shape[2:])
    
    # I_s(p_t)
    return bilinear_sampling(src, p_s)

def get_pose_mat(pose_6dof):
    ''' 
    Returns pose matrix [R|t] of shape (B,3,4) using translation 
    and euler angle values from pose_6dof (t_x, t_y, t_z, alpha, beta, gamma).
    '''
    device = pose_6dof.device
    B = pose_6dof.shape[0]

    # Extract euler angles
    alpha = torch.clip(pose_6dof[:,0,3], -torch.pi, torch.pi).view(-1,1,1,1)
    beta = torch.clip(pose_6dof[:,0,4], -torch.pi, torch.pi).view(-1,1,1,1)
    gamma = torch.clip(pose_6dof[:,0,5], -torch.pi, torch.pi).view(-1,1,1,1)
    cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
    cos_b, sin_b = torch.cos(beta), torch.sin(beta)
    cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)
    # zeros = torch.zeros((pose_6dof.shape[0],1,1,1), device=pose_6dof.device)
    # ones = torch.ones((pose_6dof.shape[0],1,1,1), device=pose_6dof.device)

    # Compute yaw, pitch, and roll
    R_z = torch.cat([
        torch.cat([cos_g, -sin_g, get_zeros((B,1,1,1), device)], axis=3),
        torch.cat([sin_g, cos_g, get_zeros((B,1,1,1), device)], axis=3),
        torch.cat([get_zeros((B,1,1,1), device), 
                   get_zeros((B,1,1,1), device), 
                   get_ones((B,1,1,1), device)], axis=3)
    ], axis=2)
    R_y = torch.cat([
        torch.cat([cos_b, get_zeros((B,1,1,1), device), sin_b], axis=3),
        torch.cat([get_zeros((B,1,1,1), device), 
                   get_ones((B,1,1,1), device), 
                   get_zeros((B,1,1,1), device)], axis=3),
        torch.cat([-sin_b, get_zeros((B,1,1,1), device), cos_b], axis=3)
    ], axis=2)
    R_x = torch.cat([
        torch.cat([get_ones((B,1,1,1), device), 
                   get_zeros((B,1,1,1), device), 
                   get_zeros((B,1,1,1), device)], axis=3),
        torch.cat([get_zeros((B,1,1,1), device), cos_a, -sin_a], axis=3),
        torch.cat([get_zeros((B,1,1,1), device), sin_a, cos_a], axis=3)
    ], axis=2)
    # Final rotation matrix
    R = (R_z @ R_y @ R_x).squeeze(1)
    # [R|t]
    t = pose_6dof[:,0,:3].unsqueeze(2)
    pose_mat = torch.cat([R,t], axis=2)
    return pose_mat

def get_p_t(img: torch.Tensor):
    ''' Get batch pixel grid for img in homogeneous coords. '''
    # x coords
    p_tx = torch.arange(img.shape[2], device=img.device).unsqueeze(1)
    p_tx = p_tx.repeat(1,img.shape[3])
    # y coords
    p_ty = torch.arange(img.shape[3], device=img.device).repeat(img.shape[2])
    p_ty = p_ty.view(img.shape[2], img.shape[3])
    # z coords
    p_tz = torch.ones_like(p_ty, device=img.device)
    # Combine into batch homogeneous coords
    p_t = torch.stack([p_ty, p_tx, p_tz], 0).unsqueeze(0)
    p_t = p_t.repeat(img.shape[0],1,1,1) 
    return p_t.to(torch.float32)

def bilinear_sampling(src, p_s):
    ''' Returns a new image by bilinear sampling values from src 
    with indices from p_s. 
    
    Args:
        src: Batch images for single source view (B,1,H_s,W_s)
        p_s: p_t's unnormalized projected coordinates (x,y) in 
            source view (B,2,H_t,W_t).
    '''
    # Normalize p_s by H,W (grid_sample expects values in [-1,1])
    p_s[:,0] = p_s[:,0] / ((p_s.shape[3]-1)/2) - 1.
    p_s[:,1] = p_s[:,1] / ((p_s.shape[2]-1)/2) - 1.

    pred = nn.functional.grid_sample(
        src, p_s.permute(0,2,3,1), align_corners=True
        )
    return pred

def compute_smooth_loss(depth:torch.Tensor, loss:nn.L1Loss):
    ''' L1 of second order gradients of depth map. 
    
    Args:
        depth: Depth map prediction (B,1,H,W)
        loss: L1 loss
    '''
    dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    dxdy = dx[:, :, 1:, :] - dx[:, :, :-1, :]
    dxdx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    dydy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    dydx = dy[:, :, :, 1:] - dy[:, :, :, :-1]
    return  loss(dxdy, torch.zeros_like(dxdy, device=depth.device)) + \
            loss(dxdx, torch.zeros_like(dxdx, device=depth.device)) + \
            loss(dydx, torch.zeros_like(dydx, device=depth.device)) + \
            loss(dydy, torch.zeros_like(dydy, device=depth.device))


# Pose Helpers

def get_src1_origin_pose(pose: torch.Tensor):
    ''' 
    Converts pose from target origin to src1 origin.

    Args:
        pose: predicted 6DOF pose with target origin (B,n_src,6)
    Returns:
        predicted matrix pose with src1 origin (B,n_src,3,4)
    '''
    device = pose_src1.device
    B = pose_src1.shape[0]
    
    # Euler to rotation matrices [R|t]
    pose_src1 = get_pose_mat(pose[:,:1])
    pose_src2 = get_pose_mat(pose[:,1:])
    # Origin pose
    pose_target = torch.eye(4, device=pose.device)[:-1].unsqueeze(0)
    pose_target = pose_target.repeat(pose.shape[0],1,1)

    # (B,3,4,4)
    poses = torch.stack([
        torch.cat([pose_src1, get_pose_pad(B, device)], dim=1),
        torch.cat([pose_target, get_pose_pad(B, device)], dim=1),
        torch.cat([pose_src2, get_pose_pad(B, device)], dim=1)
        ], dim=1)
    
    # (B,1,4,4)
    src1_pose = torch.cat([
        pose_src1, get_pose_pad(B, device)
        ], dim=1).unsqueeze(1)

    src1_origin_pose = (src1_pose @ torch.linalg.inv(poses))[:,:,:-1]
    return src1_origin_pose