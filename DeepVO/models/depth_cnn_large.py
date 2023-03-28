from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DepthCNN(nn.Module):
    ''' 
    Note: Not using multi-scale output
    '''
    def __init__(self, skip=False, alpha=10, beta=0.01):
        super().__init__()

        self.skip = skip
        self.alpha = alpha
        self.beta = beta
        
        self.enc1 = EncoderBlock(1, 32, kernel_size=(7,7))
        self.enc2 = EncoderBlock(32, 64, kernel_size=(5,5))
        self.enc3 = EncoderBlock(64, 128, kernel_size=(3,3))
        self.enc4 = EncoderBlock(128, 256, kernel_size=(3,3))
        self.enc5 = EncoderBlock(256, 512, kernel_size=(3,3))
        self.enc6 = EncoderBlock(512, 512, kernel_size=(3,3))
        self.enc7 = EncoderBlock(512, 512, kernel_size=(3,3))

        self.dec7 = DecoderBlock(512, 1024, 512, kernel_size=(3,3), skip=skip)
        self.dec6 = DecoderBlock(512, 1024, 512, kernel_size=(3,3), skip=skip)
        self.dec5 = DecoderBlock(512, 512, 256, kernel_size=(3,3), skip=skip)
        self.dec4 = DecoderBlock(256, 256, 128, kernel_size=(3,3), skip=skip)
        self.out4 = nn.Sequential(
            nn.Conv2d(128, 1, (3,3), 1, padding=1),
            nn.Sigmoid()
        )
        self.dec3 = DecoderBlock(128, 129, 64, kernel_size=(3,3), skip=skip)
        self.out3 = nn.Sequential(
            nn.Conv2d(64, 1, (3,3), 1, padding=1),
            nn.Sigmoid()
        )
        self.dec2 = DecoderBlock(64, 65, 32, kernel_size=(3,3), skip=skip)
        self.out2 = nn.Sequential(
            nn.Conv2d(32, 1, (3,3), 1, padding=1),
            nn.Sigmoid()
        )
        self.dec1 = DecoderBlock(32, 17, 16, kernel_size=(3,3), skip=skip)
        self.out1 = nn.Sequential(
            nn.Conv2d(16, 1, (3,3), 1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        
        dec7 = self.dec7(enc7, enc6)
        dec6 = self.dec6(dec7, enc5)
        dec5 = self.dec5(dec6, enc4)

        dec4 = self.dec4(dec5, enc3)
        out4 = self.alpha*self.out4(dec4) + self.beta
        out4_up = TF.resize(out4, (x.shape[2]//4, x.shape[3]//4))

        dec3 = self.dec3(dec4, enc2, out4_up)
        out3 = self.alpha*self.out3(dec3) + self.beta
        out3_up = TF.resize(out3, (x.shape[2]//2, x.shape[3]//2))

        dec2 = self.dec2(dec3, enc1, out3_up)
        out2 = self.alpha*self.out2(dec2) + self.beta
        out2_up = TF.resize(out2, (x.shape[2], x.shape[3]))

        dec1 = self.dec1(dec2, out2_up)
        out1 = self.alpha*self.out1(dec1) + self.beta
        return (out1, out2, out3, out4)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size, 
                                2, padding=(kernel_size[0]-1)//2)),
            ('norm1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size, 
                                1, padding=(kernel_size[0]-1)//2)),
            ('norm2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        return self.block(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, int_channels, out_channels, kernel_size, skip=False):
        super().__init__()
        self.skip = skip
        self.conv_transpose = nn.Sequential(OrderedDict([
            ('conv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                        stride=2, padding=1, output_padding=1)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))
        channels = out_channels if not skip else int_channels
        # padding = 0 if not pad else 1
        self.conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(channels, out_channels, kernel_size, 1, padding=1)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x, skip, disp_prev=None):
        x = self.conv_transpose(x)
        x = TF.resize(x, skip.shape[2:])
        if self.skip:
            cat = [x, skip] if disp_prev is None else [x, skip, disp_prev]
            x = torch.cat(cat, axis=1)
        return self.conv(x)