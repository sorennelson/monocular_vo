import torch
import torch.nn as nn
from collections import OrderedDict


class PoseCNN(nn.Module):
    ''' 
    Note: Not using multi-scale output
    '''
    def __init__(self, exp=False):
        super().__init__()
        self.n_src = 2
        # Encoder layers shared by pose and explainability
        self.enc = nn.Sequential(
            EncoderBlock(3, 16, kernel_size=(7,7)),
            EncoderBlock(16, 32, kernel_size=(5,5)),
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
        self.exp = nn.Identity()
        if exp:
            self.exp = nn.Sequential(
                DecoderBlock(256, 256, kernel_size=(3,3)),
                DecoderBlock(256, 128, kernel_size=(3,3)),
                DecoderBlock(128, 64, kernel_size=(3,3)),
                DecoderBlock(64, 32, kernel_size=(5,5)),
                DecoderBlock(32, 16, kernel_size=(7,7)),
                nn.Conv2d(16, self.n_src, (7,7), 1, padding=3),
                nn.Sigmoid()
            )
    
    def drop_exp(self):
        ''' Remove explainability stem for inference. '''
        self.exp = nn.Identity()

    def forward(self, target, src):
        x = torch.cat([target, src], axis=1)
        enc = self.enc(x)
        # Small constant scale from paper
        pose = 0.01 * self.pose(enc).view(-1,self.n_src,6)
        exp = self.exp(enc)
        return pose, exp

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, 
                               stride, padding=(kernel_size[0]-1)//2)),
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
                                        stride=2, padding=(kernel_size[0]-1)//2, output_padding=1)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)