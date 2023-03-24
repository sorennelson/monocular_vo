from dataset import get_training_data_loaders
from models.depth_cnn import DepthCNN
from models.pose_cnn import PoseCNN
from utils import MetricLogger, compute_smooth_loss, project_warp, get_pose_mat

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# Input
parser.add_argument('--data_path', type=str, 
                    help='Remote/local path where full dataset is persistently stored.')
parser.add_argument('--data_path_local', type=str, default='/tmp/kitti_vo_pose',
                    help='Local working dir where dataset is cached during operation.')
parser.add_argument('--n_workers', type=int, default=1, help='Number of workers.')
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

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: Performance w/o explainability
# TODO: move args to Odometry
# TODO: Verify pose MSE
# TODO: Test dataset
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
    ''' Returns the pose translation MSE as relative translation from Target->Src. 

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
    logger = MetricLogger(
        args.save_path, args.job_id, args.batch_size, args.epochs
        )
    logger.log_str(str(args))

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


if __name__=='__main__':
    main()
