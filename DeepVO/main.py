from dataset import get_training_data_loaders, get_test_data_loader
from models.depth_cnn_large import DepthCNN
from models.pose_cnn_large import PoseCNN
from utils import MetricLogger, compute_smooth_loss, projective_inverse_warp, get_src1_origin_pose

import os, time, argparse
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
# Input
parser.add_argument('--data_path', type=str, 
                    help='Remote/local path where full dataset is persistently stored.')
parser.add_argument('--data_path_local', type=str, default='/tmp/kitti_vo',
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
parser.add_argument('--lambda_s', type=float, default=0.1, help='Smooth loss scalar.')
parser.add_argument('--smooth_disp', action='store_true', 
                    help='If true applies smoothness loss to the disparity, otherwise to the depth.')
parser.add_argument('--lambda_e', type=float, default=0.0, help='Explainability loss scalar.')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate.')
parser.add_argument('--decay', type=float, default=0., help='Weight decay.')
parser.add_argument('--epochs', type=int, default=25, help='Training epochs.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
parser.add_argument('--n_src', type=int, default=2, help='Number of source images.')
parser.add_argument('--n_scales', type=int, default=1, 
                    help='Number of image scales to use when training in range [1,4].')

parser.add_argument('--evaluate', action='store_true', 
                    help='Evaluate the pretrained model stored in save_path on the test set.')
parser.add_argument('--n_workers', type=int, default=1, help='Number of workers.')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
         
        disp = depth_net(target)
        pose, exp = pose_net(target, src)

        loss = compute_multi_scale_loss(target, src, disp, pose, exp, cam)

        opt.zero_grad()
        loss.backward()
        opt.step()

        logger.update(len(target), loss.item(), 0.)
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
            
            disp = depth_net(target)
            pose, exp = pose_net(target, src)

            loss = compute_multi_scale_loss(target, src, disp, pose, exp, cam)
            pose_dist = compute_snippet_ATE(pose, pose_gt)

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

            pose_dist = compute_snippet_ATE(pose, pose_gt)

            logger.update(len(target), 0., pose_dist)
        
    logger.log_epoch_stats(train=False)


def compute_multi_scale_loss(target: torch.Tensor, 
                             src: torch.Tensor, 
                             disp: torch.Tensor, 
                             pose: torch.Tensor, 
                             exp: torch.Tensor, 
                             cam: torch.Tensor):
    h, w = target.shape[2], target.shape[3]
    loss = 0.
    for s in range(args.n_scales):
        # Reshape target and src_images to correct scale
        target_s = TF.resize(target, (h//(2**s), w//(2**s)))
        src_s = TF.resize(src, (h//(2**s), w//(2**s)))

        # update smooth weight and pass in (adjust to pass in loss scalars?)
        loss += compute_single_scale_loss(
            target_s, 
            src_s, 
            disp[s], 
            pose,
            exp[s], 
            cam[:,s], 
            args.lambda_s/(2**s), 
            args.lambda_e
        )

    return loss


def compute_single_scale_loss(target: torch.Tensor, 
                              src: torch.Tensor, 
                              disp: torch.Tensor, 
                              pose: torch.Tensor, 
                              exp: torch.Tensor, 
                              cam: torch.Tensor,
                              lambda_s: int,
                              lambda_e: int):
    ''' 
    Returns loss as in "Unsupervised Learning of Depth and Ego-Motion from Video".
    `L = L_viewsynthesis + lambda_s * L_smooth + lambda_e * L_reg.`

    Args:
        target: Target images (B,1,H,W)
        src: Source images (B,n_src,H,W)
        disp: Disparity prediction (B,1,H,W)
        pose: 6DOF pose predictions as target->src (B,n_src,6)
        exp: Explainability prediction probabilities (B,n_src*2,H,W)
        cam: Camera intrinsics (B,3,3)
        lambda_s: Smooth loss scalar
        lambda_e: Explainability loss scalar
    '''
    depth = 1./disp

    # Explainability regularizer loss
    l_e = 0.
    if args.lambda_e > 0.:
        exp_loss = nn.BCELoss()
        exp_target = torch.ones_like(exp, device=exp.device)
        l_e = exp_loss(exp, exp_target)

    # Smooth loss
    smooth_inp = disp if args.smooth_disp else depth
    l_s = compute_smooth_loss(smooth_inp, nn.L1Loss())

    # View synthesis loss per source
    view_synth_loss = nn.L1Loss(reduction='none')
    l_vs = 0.
    for i in range(src.shape[1]):
        
        proj, mask = projective_inverse_warp(src[:,i:i+1], depth, pose[:,i:i+1], cam)
        l_vsi = view_synth_loss(proj, target) * mask.unsqueeze(1)

        if lambda_e > 0.:
            # Apply explainability mask to view synthesis loss
            l_vs += torch.mean(exp[:,i:i+1]*l_vsi) 
        else:
            l_vs += torch.mean(l_vsi)

    return l_vs + lambda_s * l_s + lambda_e * l_e


def compute_snippet_ATE(pose: torch.Tensor, pose_gt: torch.Tensor):
    ''' 
    Returns the ATE of the snippet given the predicted pose (target origin) and 
    gt pose (src1 origin). Does this by converting the pose predictions from target 
    origin (src1<-target->src2) to src1 origin (src1->target->src2). Uses the pose_gt 
    to gather the scaling factor and compute final ATE.

    Args:
        pose: predicted 6DOF with target origin (B,n_src,6)
        pose_gt: gt pose matrix with src origin (B,n_src+1,3,4)
    '''
    poses = get_src1_origin_pose(pose)

    # Compute ATE
    seq_len = pose_gt.shape[1]
    ATE = 0.
    for i in range(len(poses)):
        scale = torch.sum(pose_gt[i,:,:,-1] * poses[i,:,:,-1]) 
        scale /= torch.sum(poses[i,:,:,-1] ** 2)
        samp_ATE = torch.linalg.norm((pose_gt[i,:,:,-1] - scale * poses[:,:,:,-1]).reshape(-1)) 
        samp_ATE /= seq_len
        ATE += samp_ATE
    return ATE / pose_gt.shape[0]


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
    log_job_id = args.job_id if not args.evaluate else f'{args.job_id}-test'
    logger = MetricLogger(
        args.save_path, log_job_id, args.batch_size, args.epochs
        )
    logger.log_str(str(args))

    # Load pretrained model and test
    if args.evaluate:
        test_loader = get_test_data_loader(
            args.data_path, args.data_path_local, args.n_src, args.n_workers
        )
        path = os.path.join(args.save_path, args.job_id, 'checkpoint.pth.tar')
        assert os.path.exists(path), f'Bad checkpoint path: {path}'

        pose_net = PoseCNN(args.lambda_e > 0., args.n_src)
        pose_net.load_state_dict(torch.load(path, map_location=device))
        pose_net.drop_exp()
        pose_net.to(device)

        logger.start_epoch(0)

        start = time.time()
        validate_pose(test_loader, pose_net, logger)
        logger.log_str(f'Eval time: {time.time()-start}s')

        return
    
    # Load dataloader
    train_loader, test_loader = get_training_data_loaders(
        args.data_path, args.data_path_local, args.batch_size, args.n_src, args.n_workers
    )

    # Load models
    depth_net = DepthCNN(args.skip)
    pose_net = PoseCNN(args.lambda_e > 0., args.n_src)
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
