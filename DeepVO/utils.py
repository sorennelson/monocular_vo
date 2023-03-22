import os, time
import torch


class MetricLogger:
    ''' Helper class for storing metrics and logging. '''

    def __init__(self, save_path, job_id, batch_size, epochs):
        self.reset_metrics()
        self.epoch = 0
        self.start = None
        self.batch_size = batch_size
        self.epochs = epochs
        if not os.path.isdir(os.path.join(save_path, job_id)):
            os.makedirs(os.path.join(save_path, job_id))
        self.log = open(os.path.join(save_path, job_id, 'log.txt'), 'w')
        
    def start_epoch(self, epoch):
        self.start = time.time()
        self.epoch = epoch
        self.reset_metrics()
        self.log.write(f'Training epoch {epoch}/{self.epochs} ...\n')
        self.log.flush()

    def reset_metrics(self):
        self.loss = 0.
        self.pose_dist = 0.
        self.n_examples = 0

    def update(self, batch_size, loss, pose_dist):
        self.loss += loss*batch_size
        self.pose_dist += pose_dist*batch_size
        self.n_examples += batch_size

    def log_stats(self):
        self.log.write(
            f'..... Epoch {self.epoch}/{self.epochs}, ' +
            f'iter: {self.n_examples // self.batch_size}, ' +
            f'time: {int((time.time() - self.start) // 60)}, ' +
            f'loss: {self.loss/self.n_examples:.3f}, ' +
            f'pose dist: {self.pose_dist/self.n_examples:.3f}\n')
        self.log.flush()
    
    def log_epoch_stats(self, train):
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
                f'avg train pose dist: {self.pose_dist/self.n_examples:.3f}\n')
        else:
            self.log.write(
                f'==> Validation --- ' +
                f'avg val loss: {self.loss/self.n_examples:.3f}, ' +
                f'avg val pose dist: {self.pose_dist/self.n_examples:.3f}\n\n')
        self.log.flush()


# Loss Helpers

def project_warp(src, depth, pose, cam):
    ''' Projects p_t onto source views using predicted depth and camera pose 
    then warps the projected point values using bilinear interpolation.
    '''
    # pose_mat = torch.hstack(
    #     [torch.eye(3), torch.zeros((3,1))]
    #     ).repeat(src.shape[0],1,1)
    # P = K @ [R|t]
    pose_mat = get_pose_mat(pose)
    P = torch.matmul(cam, pose_mat)
    
    # X = D @ K^-1 @ p_t
    p_t = get_p_t(src)
    cam_coords = torch.linalg.solve(
        cam, p_t.view(src.shape[0],3,-1)
        ).view(src.shape[0],3,*src.shape[2:])
    cam_coords *= depth
    # Homogenous transform
    cam_coords = torch.cat(
        [cam_coords, torch.ones((src.shape[0], 1, *cam_coords.shape[2:]))],
        dim=1)
    
    # p_s = P @ X
    p_s = torch.matmul(P, cam_coords.view(src.shape[0], 4, -1))
    # Non-homogenous transform
    p_s = torch.cat([
        p_s[:,:1] / (p_s[:,2:3] + 1e-10),
        p_s[:,1:2] / (p_s[:,2:3] + 1e-10)
    ], axis=1).view(src.shape[0],2,*src.shape[2:])
    
    # I_s(p_t)
    return bilinear_sampling(src, p_s)

def get_pose_mat(pose_6dof):
    ''' Returns pose matrix [R|t] of shape (B,3,4) using translation 
    and euler angle values from pose_6dof (t_x, t_y, t_z, alpha, beta, gamma).
    '''
    # Extract euler angles
    alpha = torch.clip(pose_6dof[:,0,3], -torch.pi, torch.pi).view(-1,1,1,1)
    beta = torch.clip(pose_6dof[:,0,4], -torch.pi, torch.pi).view(-1,1,1,1)
    gamma = torch.clip(pose_6dof[:,0,5], -torch.pi, torch.pi).view(-1,1,1,1)
    cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
    cos_b, sin_b = torch.cos(beta), torch.sin(beta)
    cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)
    zeros = torch.zeros(pose_6dof.shape[0],1,1,1)
    ones = torch.ones(pose_6dof.shape[0],1,1,1)
    # Compute yaw, pitch, and roll
    R_z = torch.cat([
        torch.cat([cos_g, -sin_g, zeros], axis=3),
        torch.cat([sin_g, cos_g, zeros], axis=3),
        torch.cat([zeros, zeros, ones], axis=3)
    ], axis=2)
    R_y = torch.cat([
        torch.cat([cos_b, zeros, sin_b], axis=3),
        torch.cat([zeros, ones, zeros], axis=3),
        torch.cat([-sin_b, zeros, cos_b], axis=3)
    ], axis=2)
    R_x = torch.cat([
        torch.cat([ones, zeros, zeros], axis=3),
        torch.cat([zeros, cos_a, -sin_a], axis=3),
        torch.cat([zeros, sin_a, cos_a], axis=3)
    ], axis=2)
    # Final rotation matrix
    R = (R_z @ R_y @ R_x).squeeze()
    # [R|t]
    t = pose_6dof[:,0,:3].unsqueeze(2)
    pose_mat = torch.cat([R,t], axis=2)
    return pose_mat

def get_p_t(src):
    ''' Get batch pixel grid in homogeneous coords.'''
    # x coords
    p_tx = torch.arange(src.shape[2]).unsqueeze(1)
    p_tx = p_tx.repeat(1,src.shape[3])
    # t coords
    p_ty = torch.arange(src.shape[3]).repeat(src.shape[2])
    p_ty = p_ty.view(src.shape[2], src.shape[3])
    # z coords
    p_tz = torch.ones_like(p_ty)
    # Combine into batch homogeneous coords
    p_t = torch.stack([p_tx, p_ty, p_tz], 0).unsqueeze(0)
    p_t = p_t.repeat(src.shape[0],1,1,1) 
    return p_t.to(torch.float32)

def bilinear_sampling(src, p_s):
    ''' Returns a new image by bilinear sampling values from src with indices from p_s. '''
    height, width = src.shape[2], src.shape[3]
    p_s = p_s.view(p_s.shape[0], 2, -1)

    # create new 0 array the size of src
    src = src.squeeze()
    pred = torch.zeros_like(src)

    tl_alpha = torch.frac(p_s)
    tl_alpha_h = tl_alpha[:,0]
    tl_alpha_w = tl_alpha[:,1]

    br = torch.ceil(p_s)
    br_alpha = br - p_s
    br_alpha_h = br_alpha[:,0]
    br_alpha_w = br_alpha[:,1]
    
    # Top left indices
    tl = torch.floor(p_s).to(torch.long).permute(0,2,1)
    # Mask out of range indices
    tl_mask = get_out_of_range_mask(height, width, tl)
    tl_valid = torch.where(tl_mask, tl, 0)
    tl_h, tl_w = tl_valid[:,:,0], tl_valid[:,:,1] 

    # Bottom right indices
    br = br.to(torch.long).permute(0,2,1)
    # Mask out of range indices
    br_mask = get_out_of_range_mask(height, width, br)
    br_valid = torch.where(br_mask, br, 0)
    br_h, br_w = br_valid[:,:,0], br_valid[:,:,1] 

    # tr/bl can be extracted from tl/br
    tr_h, tr_w = tl_h, br_w 
    bl_h, bl_w = br_h, tl_w

    for i in range(src.shape[0]):
        pred[i,tl_h[i],tl_w[i]] += src[i,tl_h[i],tl_w[i]] * tl_alpha_h[i] * tl_alpha_w[i]
        pred[i,tr_h[i],tr_w[i]] += src[i,tr_h[i],tr_w[i]] * tl_alpha_h[i] * br_alpha_w[i]
        pred[i,bl_h[i],bl_w[i]] += src[i,bl_h[i],bl_w[i]] * br_alpha_h[i] * tl_alpha_w[i]
        pred[i,br_h[i],br_w[i]] += src[i,br_h[i],br_w[i]] * br_alpha_h[i] * br_alpha_w[i]

    return pred

def get_out_of_range_mask(height, width, grid):
    ''' Returns a mask with the same shape as grid, where: 
    mask[i,j]==[1,1] if grid[i,j,0] < height and grid[i,j,1] < width and grid[i,j] >= 0,  
    else mask[i,j]==[0,0].
    '''
    height_mask = (grid[:,:,:1] < height) * (grid[:,:,:1] >= 0) 
    width_mask = (grid[:,:,1:] < width) * (grid[:,:,1:] >= 0) 
    return (height_mask*width_mask).repeat(1,1,2)

def compute_smooth_loss(depth, loss):
    ''' L1 of second order gradients of depth map. '''
    dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    dxdy = dx[:, :, 1:, :] - dx[:, :, :-1, :]
    dxdx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    dydy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    dydx = dy[:, :, :, 1:] - dy[:, :, :, :-1]
    return  loss(dxdy, torch.zeros_like(dxdy)) + \
            loss(dxdx, torch.zeros_like(dxdx)) + \
            loss(dydx, torch.zeros_like(dydx)) + \
            loss(dydy, torch.zeros_like(dydy))
