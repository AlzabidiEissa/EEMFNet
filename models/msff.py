import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .coordatt import CoordAtt



class SA(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(SA, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(in_channel)
        self.act1 = nn.ReLU(inplace=True)
        self.attn = CoordAtt(in_channel, in_channel)
        
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(in_channel)
        self.act2 = nn.ReLU(inplace=True)
        

        self.conv3 = nn.Conv2d(in_channel, 2*in_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(2*in_channel)
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_conv = self.conv1(x)
        x_conv = self.bn1(x_conv)
        x_conv = self.act1(x_conv)        
        
        x_att = self.attn(x)
        
        out1 = x_conv * x_att
        
        out2 = self.act2(self.bn2(self.conv2(out1))) #in_channel

        out3 = self.bn3(self.conv3(out2))
        w, b = out3[:, :self.in_channel, :, :], out3[:, self.in_channel:, :, :]
        out3 = self.act3(w * out2 + b)
        
        return out3

class LearnableEdgeDetection(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LearnableEdgeDetection, self).__init__()
        self.edge_conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    
    def forward(self, x):

        freq = torch.fft.fft2(x, norm='ortho')
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        # Combine real and imaginary parts
        freq2 = torch.stack([freq.real, freq.imag], -1)
        # Compute the magnitude of the frequency components
        freq_magnitude = torch.sqrt(freq2[..., 0]**2 + freq2[..., 1]**2)  
        
        return torch.sigmoid(self.edge_conv(freq_magnitude))  # Apply sigmoid for edge probability map


class MSFF(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(MSFF, self).__init__()

        # self.in_ch1, self.in_ch2, self.in_ch3 = [2 * x for x in in_channels] 
        self.in_ch1, self.in_ch2, self.in_ch3 = in_channels 

        self.blk1 = SA(self.in_ch1)
        self.blk2 = SA(self.in_ch2)
        self.blk3 = SA(self.in_ch3)


        # self.edge1 = LearnableEdgeDetection(input_channels=self.in_ch1//2, output_channels=1)
        # self.edge2 = LearnableEdgeDetection(input_channels=self.in_ch2//2, output_channels=1)
        # self.edge3 = LearnableEdgeDetection(input_channels=self.in_ch3//2, output_channels=1)

        self.edge1 = LearnableEdgeDetection(input_channels=self.in_ch1, output_channels=1)
        self.edge2 = LearnableEdgeDetection(input_channels=self.in_ch2, output_channels=1)
        self.edge3 = LearnableEdgeDetection(input_channels=self.in_ch3, output_channels=1)

        # # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.upconv32 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(self.in_ch3//2, self.in_ch2//2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(self.in_ch3, self.in_ch2, kernel_size=2, stride=2),
            norm_layer(self.in_ch2),
            nn.ReLU(inplace=True)            
        )
        self.upconv21 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(self.in_ch2//2, self.in_ch1//2, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(self.in_ch2, self.in_ch1, kernel_size=2, stride=2),
            norm_layer(self.in_ch1),
            nn.ReLU(inplace=True)            
        )

    def forward(self, features):
        # features = [level1, level2, level3]
        f1, f2, f3 = features 

        # # score map
        # m3 = f3[:,self.in_ch3//2:,...].mean(dim=1, keepdim=True)
        # m2 = f2[:,self.in_ch2//2:,...].mean(dim=1, keepdim=True)
        # m1 = f1[:,self.in_ch1//2:,...].mean(dim=1, keepdim=True)


        m1 = self.edge1(f1)
        m2 = self.edge2(f2)       
        m3 = self.edge3(f3)
        
        # MSFF Module
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)

        f2_f = f2_k + self.upconv32(f3_k)
        f1_f = f1_k + self.upconv21(f2_f)


        # return [f1_f, f2_f, f3_k]

        # spatial attention
          
        m1_weight = m1 * self.upsample2(m2) * self.upsample4(m3)
        # print('self.in_ch//2',self.in_ch3//2, self.in_ch2//2, self.in_ch1//2)
 
        m2_weight = m2 * self.upsample2(m3)
        m3_weight = m3    

        f1_out = f1_f * m1_weight
        f2_out = f2_f * m2_weight
        f3_out = f3_k * m3_weight
        
        return [f1_out, f2_out, f3_out]
