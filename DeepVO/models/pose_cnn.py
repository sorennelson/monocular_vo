import torch
import torch.nn as nn
from collections import OrderedDict


class PoseCNN(nn.Module):
    ''' 
    Note: Not using multi-scale output
    Note: No explainability output yet
    '''
    def __init__(self, exp=False):
        super().__init__()
        self.n_src = 2
        # Encoder layers shared by pose and explainability
        self.enc = nn.Sequential(
            EncoderBlock(3, 16, kernel_size=(3,3)), # kernel=7 in paper
            EncoderBlock(16, 32, kernel_size=(3,3)), # kernel=5 in paper
            EncoderBlock(32, 64, kernel_size=(3,3)),
            EncoderBlock(64, 128, kernel_size=(3,3)),
            EncoderBlock(128, 256, kernel_size=(3,3))
        )
        # Pose prediction head
        self.pose = nn.Sequential(
            EncoderBlock(256, 256, kernel_size=(3,3), stride=1),
            EncoderBlock(256, 256, kernel_size=(3,3), stride=1),
            nn.Conv2d(256, 6*self.n_src, (1,1), 1),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Explainability Mask head
        self.exp = None
        if exp:
            self.exp = nn.Sequential(
                DecoderBlock(256, 256, kernel_size=(3,3)),
                DecoderBlock(256, 128, kernel_size=(3,3)),
                DecoderBlock(128, 64, kernel_size=(3,3)),
                DecoderBlock(64, 32, kernel_size=(3,3)),
                DecoderBlock(32, 16, kernel_size=(3,3)),
                nn.Conv2d(16, 2*self.n_src, (1,1), 1),
            )


    def forward(self, target, src):
        x = torch.cat([target, src], axis=1)
        enc = self.enc(x)
        # Small constant scale from paper
        pose = 0.01 * self.pose(enc).view(-1,self.n_src,6)

        exp = None
        if self.exp:
            exp = self.exp(enc)
            # Reshape to [B*n_src, 2, H, W]
            exp = exp.view(-1, 2, *exp.shape[2:])
            # Softmax explanation per source view
            exp = nn.functional.softmax(exp, dim=1)
            # Reshape to [B, 2*n_src, H, W]
            exp = exp.view(src.shape[0], 2*self.n_src, *exp.shape[2:])

        return pose, exp

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, 
                               stride, padding=1)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                        stride=2, padding=1, output_padding=1)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)