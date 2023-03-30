import torch
import torch.nn as nn
from collections import OrderedDict


class PoseCNN(nn.Module):
    def __init__(self, exp=False, n_src=2):
        super().__init__()
        self.n_src = n_src
        # Encoder layers shared by pose and explainability
        self.enc = nn.Sequential(
            EncoderBlock(self.n_src + 1, 16, kernel_size=(7,7)),
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
            # nn.AdaptiveAvgPool2d((1,1))
        )
        # Explainability Mask head
        self.exp = exp
        if exp:
            self.exp5 = DecoderBlock(256, 256, kernel_size=(3,3))
            self.exp4 = DecoderBlock(256, 128, kernel_size=(3,3))
            self.out4 = nn.Sequential(
                nn.Conv2d(128, self.n_src, (3,3), padding=1),
                nn.Sigmoid()
            )
            self.exp3 = DecoderBlock(128, 64, kernel_size=(3,3))
            self.out3 = nn.Sequential(
                nn.Conv2d(64, self.n_src, (3,3), padding=1),
                nn.Sigmoid()
            )
            self.exp2 = DecoderBlock(64, 32, kernel_size=(3,3))
            self.out2 = nn.Sequential(
                nn.Conv2d(32, self.n_src, (3,3), padding=1),
                nn.Sigmoid()
            )
            self.exp1 = DecoderBlock(32, 16, kernel_size=(3,3))
            self.out1 = nn.Sequential(
                nn.Conv2d(16, self.n_src, (3,3), padding=1),
                nn.Sigmoid()
            )
    
    def drop_exp(self):
        ''' Remove explainability stem for inference. '''
        self.exp = False

    def forward(self, target, src):
        x = torch.cat([target, src], axis=1)
        enc = self.enc(x)

        # Small constant scale from paper
        # pose = 0.01 * self.pose(enc).view(-1,self.n_src,6)
        pose = 0.01 * self.pose(enc).mean((2,3)).view(-1,self.n_src,6)
        
        exp = (None, None, None, None)
        if self.exp:
            exp5 = self.exp5(enc)
            exp4 = self.exp4(exp5)
            out4 = self.out4(exp4)
            exp3 = self.exp3(exp4)
            out3 = self.out3(exp3)
            exp2 = self.exp2(exp3)
            out2 = self.out2(exp2)
            exp1 = self.exp1(exp2)
            out1 = self.out1(exp1)
            
            exp = (out1, out2, out3, out4)

        return pose, exp

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, 
                               stride, padding=(kernel_size[0]-1)//2)),
            # ('norm', nn.BatchNorm2d(out_channels)),
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
            # ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)