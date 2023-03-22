import torch
import torch.nn as nn
from collections import OrderedDict


class PoseCNN(nn.Module):
    ''' 
    Note: Not using multi-scale output
    Note: No explainability output yet
    '''
    def __init__(self):
        super().__init__()
        self.n_src = 2
        # Encoder layers shared by pose and explainability
        self.enc = nn.Sequential(
            EncoderBlock(3, 16, kernel_size=(3,3)), # kernel=7 in paper
            EncoderBlock(16, 32, kernel_size=(3,3)), # kernel=5 in paper
            EncoderBlock(32, 64, kernel_size=(3,3)),
            EncoderBlock(64, 128, kernel_size=(3,3)),
            EncoderBlock(128, 256, kernel_size=(3,3), stride=1)
        )
        # Pose prediction head
        self.pose = nn.Sequential(
            EncoderBlock(256, 256, kernel_size=(3,3), stride=1),
            EncoderBlock(256, 256, kernel_size=(3,3), stride=1),
            nn.Conv2d(256, 6*self.n_src, (1,1), 1),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, target, src):
        x = torch.cat([target, src], axis=1)
        enc = self.enc(x)
        # small constant scale from paper
        pose = 0.01 * self.pose(enc).view(-1,self.n_src,6)
        return pose

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)