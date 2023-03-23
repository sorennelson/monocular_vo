from dataset import get_training_data_loaders
from depth_cnn import DepthCNN
from pose_cnn import PoseCNN
from utils import MetricLogger, compute_smooth_loss, project_warp

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# Input
parser.add_argument('--data_path', type=str, 
                    help='Remote/local path where full dataset is persistently stored.')
parser.add_argument('--data_path_local', type=str, default='/tmp/kitti_vo_stream',
                    help='Local working dir where dataset is cached during operation.')
parser.add_argument('--n_workers', type=int, default=1, help='Number of workers.')
# Output
parser.add_argument('--job_id', type=str, help='Job identifier.')
parser.add_argument('--save_path', type=str, default='./checkpoints/', 
                    help='Path to model saving and logging directory.')
parser.add_argument('--log_frequency', type=int, default=100, 
                    help='Number of batches between logging.')
# Hyperparameters
parser.add_argument('--skip', action='store_true', 
                    help='Whether to use skip connections in Depth network.')
parser.add_argument('--lambda_s', type=float, default=0.5, help='Smooth loss scalar.')
parser.add_argument('--epochs', type=int, default=2, help='Training epochs.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate.')
parser.add_argument('--decay', type=float, default=0.05, help='Weight decay.')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(data_loader:DataLoader, depth_net:nn.Module, pose_net:nn.Module, 
          opt:torch.optim.Optimizer, logger:MetricLogger):
    depth_net.train()
    pose_net.train()

    for i, (target, src, cam) in enumerate(data_loader):
        print(target.shape, src.shape, cam.shape)
        target = target.to(device)
        src = src.to(device)
        cam = cam.to(device)
         
        depth = depth_net(target)
        pose = pose_net(target, src)
        loss = compute_loss(target, src, depth, pose, cam)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # TODO: Pose dist
        logger.update(len(target), loss.item(), 0.)
        if (i+1)%args.log_frequency == 0:
            logger.log_stats()

    logger.log_epoch_stats(train=True)


def validate(data_loader:DataLoader, depth_net:DepthCNN, pose_net:PoseCNN, 
             logger:MetricLogger):
    depth_net.eval()
    pose_net.eval()

    with torch.no_grad():
        for i, (target, src, cam) in enumerate(data_loader):
            target = target.to(device)
            src = src.to(device)
            cam = cam.to(device)
            
            depth = depth_net(target)
            pose = pose_net(target, src)
            loss = compute_loss(target, src, depth, pose, cam)

            # TODO: Pose dist
            logger.update(len(target), loss.item(), 0.)
        
    logger.log_epoch_stats(train=False)


def compute_loss(target, src, depth, pose, cam):
    ''' Returns loss as in "Unsupervised Learning of Depth and Ego-Motion from Video".
    `L = L_viewsynthesis + lambda_s * L_smooth.`
    Currently does not use Explainabilty prediction so this is ommited from the loss.
    '''
    # Smooth loss
    smooth_loss = nn.L1Loss()
    l_s = compute_smooth_loss(depth, smooth_loss)

    # View synthesis loss per source
    view_synth_loss = nn.L1Loss()
    proj_1 = project_warp(src[:,:1], depth, pose[:,:1], cam).unsqueeze(1)
    proj_2 = project_warp(src[:,1:], depth, pose[:,1:], cam).unsqueeze(1)
    l_vs = view_synth_loss(proj_1, target) + view_synth_loss(proj_2, target)

    return l_vs + args.lambda_s * l_s

def get_params(model:nn.Module, weight_decay=0.05):
    ''' 
    Return params for optimizer. Applies weight decay to non 
    normalization/bias parameters.
    '''
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if 'norm' in name.lower() or 'bias' in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def main():
    # Load dataloader
    train_loader, test_loader = get_training_data_loaders(
        args.data_path, args.data_path_local, args.batch_size, args.n_workers
        )
    
    # Load models
    depth_net = DepthCNN(args.skip)
    pose_net = PoseCNN()
    depth_net = depth_net.to(device)
    pose_net = pose_net.to(device)

    # Setup optimizer
    params = get_params(depth_net, args.decay)
    params.extend(get_params(pose_net, args.decay))
    opt = torch.optim.Adam(params, lr=args.learning_rate)

    logger = MetricLogger(
        args.save_path, args.job_id, args.batch_size, args.epochs
        )
    
    for epoch in range(1,args.epochs+1):
        logger.start_epoch(epoch)
        train(train_loader, depth_net, pose_net, opt, logger)
        logger.reset_metrics()
        validate(test_loader, depth_net, pose_net, logger)


if __name__=='__main__':
    main()
