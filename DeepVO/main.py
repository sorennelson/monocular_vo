from dataset import get_training_data_loaders, get_test_data_loader
from models.depth_cnn_large import DepthCNN
from models.pose_cnn_large import PoseCNN
from utils import MetricLogger, compute_smooth_loss, project_warp, get_pose_mat, get_src1_origin_pose

import os, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# Input
parser.add_argument('--data_path', type=str, 
                    help='Remote/local path where full dataset is persistently stored.')
parser.add_argument('--data_path_local', type=str, default='/tmp/kitti_vo_pose',
                    help='Local working dir where dataset is cached during operation.')
# Output
parser.add_argument('--job_id', type=str, help='Job identifier.')
parser.add_argument('--save_path', type=str, default='./checkpoints/', 
                    help='Path to model saving and logging directory.')
parser.add_argument('--log_frequency', type=int, default=250, 
                    help='Number of batches between logging.')
# Hyperparameters
parser.add_argument('--skip', action='store_true', 
                    help='Whether to use skip connections in Depth network.')
parser.add_argument('--lambda_s', type=float, default=0.5, help='Smooth loss scalar.')
parser.add_argument('--lambda_e', type=float, default=0.2, help='Explainability loss scalar.')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate.')
parser.add_argument('--decay', type=float, default=0.05, help='Weight decay.')
parser.add_argument('--epochs', type=int, default=10, help='Training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')

parser.add_argument('--evaluate', action='store_true', 
                    help='Evaluate the pretrained model stored in save_path on the test set.')
parser.add_argument('--n_workers', type=int, default=1, help='Number of workers.')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: move args to Odometry
# TODO: README

def train(data_loader:DataLoader, 
          depth_net:nn.Module, 
          pose_net:nn.Module, 
          opt:torch.optim.Optimizer, 
          logger:MetricLogger):
    
    depth_net.train()
    pose_net.train()

    for i, (target, src, cam, pose_gt) in enumerate(data_loader):
        target = target.to(device)
        src = src.to(device)
        cam = cam.to(device)
        pose_gt = pose_gt.to(device)
         
        depth = depth_net(target)
        pose, exp = pose_net(target, src)

        loss = compute_loss(target, src, depth, pose, exp, cam)
        pose_dist = compute_pose_metrics(pose, pose_gt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        logger.update(len(target), loss.item(), pose_dist)
        if (i+1)%args.log_frequency == 0:
            logger.log_stats()

    logger.log_epoch_stats(train=True)


def validate(data_loader:DataLoader, 
             depth_net:DepthCNN, 
             pose_net:PoseCNN, 
             logger:MetricLogger):
    
    depth_net.eval()
    pose_net.eval()

    with torch.no_grad():
        for i, (target, src, cam, pose_gt) in enumerate(data_loader):
            target = target.to(device)
            src = src.to(device)
            cam = cam.to(device)
            pose_gt = pose_gt.to(device)
            
            depth = depth_net(target)
            pose, exp = pose_net(target, src)

            loss = compute_loss(target, src, depth, pose, exp, cam)
            pose_dist = compute_pose_metrics(pose, pose_gt)

            logger.update(len(target), loss.item(), pose_dist)
        
    logger.log_epoch_stats(train=False)


def validate_pose(data_loader:DataLoader, pose_net:PoseCNN, logger:MetricLogger):
    pose_net.eval()

    with torch.no_grad():
        for i, (target, src, cam, pose_gt) in enumerate(data_loader):
            target = target.to(device)
            src = src.to(device)
            cam = cam.to(device)
            pose_gt = pose_gt.to(device)

            pose, _ = pose_net(target, src)

            pose_dist = compute_sequence_ATE(pose, pose_gt)

            logger.update(len(target), 0., pose_dist)
        
    logger.log_epoch_stats(train=False)

# def validate_global(data_loader:DataLoader, pose_net:PoseCNN, logger:MetricLogger):
    
#     # Indexes may be out of order so loop through to grab sequences and range of indexes
#     seq_max = {}
#     for i, (_, _, _, (pose_gt, pose_idx)) in enumerate(data_loader):
#         for b in range(len(pose_idx)):
#             for i in range(len(pose_idx[b])):
#                 seq = pose_idx[b,i,0].item()
#                 m = seq_max[seq] if seq in seq_max else 0
#                 seq_max[seq] = max(pose_idx[b,i,1].item(), m)
    
#     # Separate into src_1 and src_2 poses
#     seq_poses_1 = {s:torch.zeros((m+1,12)) for s,m in seq_max.items()}
#     seq_poses_2 = {s:torch.zeros((m+1,12)) for s,m in seq_max.items()}
#     seq_gt = {s:torch.zeros((m+1,12)) for s,m in seq_max.items()}

#     pose_net.eval()
#     with torch.no_grad():
#         for i, (target, src, _, pose_gt) in enumerate(data_loader):
#             target = target.to(device)
#             src = src.to(device)
#             pose_gt, pose_idx = pose_gt
#             pose_gt = pose_gt.to(device)
#             pose_idx = pose_idx.to(device)
            
#             # Predict pose
#             pose, _ = pose_net(target, src)

#             # Euler to rotation matrix [R|t] to match gt
#             pose_src1 = get_pose_mat(pose[:,:1]).view(-1,12)
#             pose_src2 = get_pose_mat(pose[:,1:]).view(-1,12)

#             for i in range(len(pose_idx)):
#                 # seq consistent for both source views
#                 seq = pose_idx[i,0,0].item()
#                 seq_poses_1[seq][pose_idx[i,0,1]] = pose_src1[i]
#                 seq_poses_2[seq][pose_idx[i,2,1]] = pose_src2[i]

#                 # TODO: Multiply seq_poses_2 by target

#                 seq_gt[seq][pose_idx[i,0,1]] = pose_gt[i,0].view(12)
#                 seq_gt[seq][pose_idx[i,2,1]] = pose_gt[i,2].view(12)

#     torch.save(seq_poses_1, os.path.join(args.save_path, args.job_id, f'pose_preds-1.pt'))
#     torch.save(seq_poses_2, os.path.join(args.save_path, args.job_id, f'pose_preds-2.pt'))  

#     # Convert each pose to global (only applied to src 2 for now)
#     for seq, poses in seq_poses_2.items():
#         errors = []
#         # Start from third frame with src2 so insert in gt pose for 
#         # frames 0 and 1 to get global pose
#         poses[:2] = seq_gt[seq][:2]

#         for i, pose in enumerate(poses):
#             if i < 2: continue

#             # poses[:,:,-1] -= pose_src1[:,-1]
#             # poses = torch.linalg.inv(pose_src1[:,:3]) @ poses


#             # Extract 
#             # t, R = pose.view(3,4)[:,-1:], pose.view(3,4)[:,:-1]
#             pose_curr = torch.cat([
#                 pose.view(3,4), torch.tensor([[0,0,0,1]], device=device)
#                 ], axis=0)
#             pose_prev = torch.cat([
#                 poses[i-1].view(3,4), torch.tensor([[0,0,0,1]])
#                 ], axis=0)
#             # t_prev, R_prev = pose_prev[:,-1:], pose_prev[:,:-1]
            
#             # t = (t_prev + (R_prev @ t)).squeeze(-1)
#             # R = R_prev @ R

#             pose_curr = pose_prev @ torch.linalg.inv(pose_curr)
#             R, t = pose_curr[:-1,:-1], pose_curr[:-1,-1]

#             # Optimize scale
#             t_gt = seq_gt[seq][i].view(3,4)[:,-1]
#             scale = torch.sum(t_gt * t)/torch.sum(t ** 2)
#             t *= scale
#             print(scale, t, t_gt)

#             poses[i] = torch.cat([R, t.unsqueeze(-1)], dim=-1).view(12)

#             errors.append(torch.linalg.norm(t - t_gt))
#             if i == 20:
#                 break

#         # Output sequence MSE
#         seq_mse = torch.mean(torch.tensor(errors))
#         logger.log_str(f'Seq {seq}: MSE {seq_mse.item()}')
        

#     # print(seq_poses_2[10][:3])
#     # print(seq_poses_2[10][-3:])


def compute_loss(target: torch.Tensor, 
                 src: torch.Tensor, 
                 depth: torch.Tensor, 
                 pose: torch.Tensor, 
                 exp: torch.Tensor, 
                 cam: torch.Tensor):
    ''' 
    Returns loss as in "Unsupervised Learning of Depth and Ego-Motion from Video".
    `L = L_viewsynthesis + lambda_s * L_smooth + lambda_e * L_reg.`

    Args:
        target: Target images (B,1,H,W)
        src: Source images (B,n_src,H,W)
        depth: Depth map prediction (B,1,H,W)
        pose: 6DOF pose predictions as target->src (B,n_src,6)
        exp: Explainability prediction probabilities (B,n_src*2,H,W)
        cam: Camera intrinsics (B,3,3)
    '''
    # Explainability regularizer loss
    l_e = 0.
    if args.lambda_e > 0.:
        exp_loss = nn.CrossEntropyLoss()
        # Reshape to [B*n_src, 2, H, W]
        exp = exp.view(-1, 2, *exp.shape[2:])
        exp_target = torch.ones((exp.shape[0], *exp.shape[2:]), 
                                dtype=torch.long, device=exp.device)
        l_e = exp_loss(exp, exp_target)
        # Grab explainability prediction
        exp = exp[:, 1:]
        # Reshape to [B, n_src, H, W]
        exp = exp.view(-1, 2, *exp.shape[2:])

    # Smooth loss
    l_s = compute_smooth_loss(depth, nn.L1Loss())

    # View synthesis loss per source
    view_synth_loss = nn.L1Loss(reduction='none')
    proj_1 = project_warp(src[:,:1], depth, pose[:,:1], cam)
    proj_2 = project_warp(src[:,1:], depth, pose[:,1:], cam)
    l_vs1 = view_synth_loss(proj_1, target) 
    l_vs2 = view_synth_loss(proj_2, target)
    if args.lambda_e > 0.:
        # Apply explainability mask to view synthesis loss
        l_vs = torch.mean(exp[:,:1]*l_vs1) + torch.mean(exp[:,1:]*l_vs2)
    else:
        l_vs = torch.mean(l_vs1) + torch.mean(l_vs2)
    return l_vs + args.lambda_s * l_s + args.lambda_e * l_e

def compute_pose_metrics(pose_pred, pose_gt):
    '''
      Returns the pose translation MSE as relative translation from 
      Target->Src (target origin). 

    Args:
        pose_pred: 6DOF pose prediction per source view (B,n_src,6) 
                    (tx, ty, tz, alpha, beta, gamma)
        pose_gt: Ground truth pose matrix [R|t] (B,n_src,3,4)
    '''
    # Euler to rotation matrix [R|t] to match gt
    pose_pred_src1 = get_pose_mat(pose_pred[:,:1])
    pose_pred_src2 = get_pose_mat(pose_pred[:,1:])
    pose_pred = torch.stack([pose_pred_src1, pose_pred_src2], dim=1)
    # B,2,3,4 -> 2B,3,4 (2 comes from 2 src views)
    pose_pred = pose_pred.view(-1, 3, 4) 
    pose_gt = pose_gt.view(-1, 3, 4)

    # Translation mse
    scale = torch.sum(pose_gt[:,:,-1] * pose_pred[:,:,-1], dim=1)
    scale /= torch.sum(pose_pred[:,:,-1] ** 2, dim=1)
    scale = scale.unsqueeze(1)
    alignment_error = pose_pred[:,:,-1] * scale - pose_gt[:,:,-1]
    mse = torch.sum(alignment_error ** 2)/len(pose_pred)

    return mse

def compute_sequence_ATE(pose: torch.Tensor, pose_gt: torch.Tensor):
    ''' 
    Returns the ATE of the sequence given the predicted pose (target origin) and 
    gt pose (src1 origin). Does this by converting the pose predictions from target 
    origin (src1<-target->src2) to src1 origin (src1->target->src2). Uses the pose_gt 
    to gather the scaling factor and compute final ATE.

    Args:
        pose: predicted 6DOF with target origin (B,2,6)
        pose_gt: gt pose matrix with src origin (B,3,3,4)
    '''
    poses = get_src1_origin_pose(pose)

    # Compute ATE
    scale = torch.sum(pose_gt[0,:,:,-1] * poses[:,:,-1]) / torch.sum(poses[:,:,-1] ** 2)
    pose_dist = torch.linalg.norm((pose_gt[0,:,:,-1] - scale * poses[:,:,-1]).reshape(-1)) 
    pose_dist /= poses.shape[0]
    return pose_dist


def get_params(model:nn.Module, weight_decay=0.05):
    ''' Return params for optimizer. Applies weight decay to non norm/bias parameters. '''
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if 'norm' in name.lower() or 'bias' in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def main():
    log_job_id = args.job_id if not args.evaluate else f'{args.job_id}-test_src1origin'
    logger = MetricLogger(
        args.save_path, log_job_id, args.batch_size, args.epochs
        )
    logger.log_str(str(args))

    # Load pretrained model and test
    if args.evaluate:
        test_loader = get_test_data_loader(
            args.data_path, args.data_path_local, args.batch_size, args.n_workers
        )
        path = os.path.join(args.save_path, args.job_id, 'checkpoint.pth.tar')
        assert os.path.exists(path), f'Bad checkpoint path: {path}'

        pose_net = PoseCNN(args.lambda_e > 0.)
        pose_net.load_state_dict(torch.load(path, map_location=device))
        pose_net.drop_exp()
        pose_net.to(device)

        logger.start_epoch(0)
        validate_pose(test_loader, pose_net, logger)
        # validate_global(test_loader, pose_net, logger)
        return
    
    # Load dataloader
    train_loader, test_loader = get_training_data_loaders(
        args.data_path, args.data_path_local, args.batch_size, args.n_workers
    )

    # Load models
    depth_net = DepthCNN(args.skip)
    pose_net = PoseCNN(args.lambda_e > 0.)
    depth_net = depth_net.to(device)
    pose_net = pose_net.to(device)

    # Setup optimizer
    params = get_params(depth_net, args.decay)
    params.extend(get_params(pose_net, args.decay))
    opt = torch.optim.Adam(params, lr=args.learning_rate)
    
    for epoch in range(1,args.epochs+1):
        logger.start_epoch(epoch)
        train(train_loader, depth_net, pose_net, opt, logger)
        logger.reset_metrics()
        validate(test_loader, depth_net, pose_net, logger)

        # Save Pose model
        torch.save(pose_net.state_dict(), 
                   os.path.join(args.save_path, args.job_id, 'checkpoint.pth.tar'))


if __name__=='__main__':
    main()
