from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision


class DepthCNN(nn.Module):
    ''' 
    Note: paper has multi-scale output, here I'm currently just using standard U-Net.
    '''
    def __init__(self, skip=False, alpha=10, beta=0.1):
        super().__init__()

        self.skip = skip
        self.alpha = alpha
        self.beta = beta
        
        self.enc1 = EncoderBlock(1, 32, kernel_size=(3,3)) # kernel=7 in paper
        self.enc2 = EncoderBlock(32, 64, kernel_size=(3,3)) # kernel=5 in paper
        self.enc3 = EncoderBlock(64, 128, kernel_size=(3,3))
        self.enc4 = EncoderBlock(128, 256, kernel_size=(3,3))
        # 3x512 in paper
        # 2x512 in paper
        self.dec4 = DecoderBlock(256, 128, kernel_size=(3,3), skip=skip)
        self.dec3 = DecoderBlock(128, 64, kernel_size=(3,3), skip=skip)
        self.dec2 = DecoderBlock(64, 32, kernel_size=(3,3), skip=skip)
        self.dec1 = DecoderBlock(32, 16, kernel_size=(3,3), skip=skip, pad=True)
        self.out = nn.Sequential(
            nn.Conv2d(16, 1, (3,3), 1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        dec4 = self.dec4(enc4, enc3)
        dec3 = self.dec3(dec4, enc2)
        dec2 = self.dec2(dec3, enc1)
        # Don't skip input in paper
        dec1 = self.dec1(dec2, x)
        out = self.out(dec1)
        out = 1. / (self.alpha*out+ self.beta)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size, 2)),
            ('norm1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size, 1)),
            ('norm2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, skip=False, pad=False):
        super().__init__()
        self.skip = skip
        self.conv_transpose = nn.Sequential(OrderedDict([
            ('conv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        channels = out_channels if not skip else out_channels*2
        padding = 0 if not pad else 1
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(channels, out_channels, kernel_size, 1, padding=padding)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x, skip):
        x = self.conv_transpose(x)
        x = torchvision.transforms.functional.resize(x, skip.shape[2:])
        if self.skip:
            x = torch.cat([x, skip], axis=1)
        return self.conv(x)