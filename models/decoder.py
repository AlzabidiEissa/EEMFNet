import torch
import torch.nn as nn


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.BatchNorm2d):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            # nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            norm_layer(out_channel),
            nn.ReLU(inplace=True) 
        )

    def forward(self, x):
        return self.blk(x)

class DBBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.BatchNorm2d):
        super(DBBlock, self).__init__()

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel )
        self.depthwise_norm = norm_layer(in_channel)
        self.depthwise_activation = nn.LeakyReLU(0.01)

        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.norm1 = norm_layer(out_channel)
        self.dropout = nn.Dropout2d(p=0.3)
        self.activation1 = nn.LeakyReLU(0.01)

        # Additional convolution
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        # Batch normalization, dropout, and activation
        self.norm2 = norm_layer(out_channel)
        self.activation2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Depthwise convolution with normalization and activation
        x = self.depthwise_conv(x)
        x = self.depthwise_norm(x)
        x = self.depthwise_activation(x)  # Apply activation after depthwise convolution

        # Pointwise convolution
        x = self.pointwise_conv(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.activation1(x)

        # Additional convolution
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        return x




class Decoder(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        self.in_ch1, self.in_ch2, self.in_ch3, self.in_ch4, self.in_ch5 = in_channels
        
        self.upconv3 = UpConvBlock(self.in_ch5, self.in_ch5//2)
        self.db3 = DBBlock(self.in_ch5//2 + self.in_ch4, self.in_ch4)

        self.upconv2 = UpConvBlock(self.in_ch4, self.in_ch4//2)
        self.db2 = DBBlock(self.in_ch4//2 + self.in_ch3, self.in_ch3)

        self.upconv1 = UpConvBlock(self.in_ch3, self.in_ch3//2)
        self.db1 = DBBlock(self.in_ch3//2 + self.in_ch2, self.in_ch2)

        self.upconv0 = UpConvBlock(self.in_ch2, self.in_ch2//2)
        self.db0 = DBBlock(self.in_ch2//2 + self.in_ch1, self.in_ch1)

        self.upconv00 = UpConvBlock(self.in_ch1, 48)
        self.db00 = DBBlock(48, 24)

        self.final_out = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            norm_layer(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 2, kernel_size=3, padding=1),
            #nn.Sigmoid(),
        )
    

    def forward(self, encoder_output, concat_features):
        # concat_features = [level0, level1, level2, level3]
        f0, f1, f2, f3 = concat_features
        
        # 512 x 8 x 8 -> 512 x 16 x 16
        x_up3 = self.upconv3(encoder_output)
        x_up3 = torch.cat([x_up3, f3], dim=1)  
        x_up3 = self.db3(x_up3)
        
        # 512 x 16 x 16 -> 256 x 32 x 32
        x_up2 = self.upconv2(x_up3)
        x_up2 = torch.cat([x_up2, f2], dim=1) 
        x_up2 = self.db2(x_up2)

        # 256 x 32 x 32 -> 128 x 64 x 64
        x_up1 = self.upconv1(x_up2)
        x_up1 = torch.cat([x_up1, f1], dim=1) 
        x_up1 = self.db1(x_up1)
        

        # 128 x 64 x 64 -> 96 x 128 x 128
        x_up0 = self.upconv0(x_up1)
        # f0 = self.conv(f0)
        # x_up2mask = torch.cat([x_up0, f0], dim=1)  
        x_up0 = torch.cat([x_up0, f0], dim=1)  
        x_up0 = self.db0(x_up0)

        
        # 96 x 128 x 128 -> 48 x 256 x 256
        # x_mask = self.upconv2mask(x_up2mask)  
        x_up00 = self.upconv00(x_up0)
        x_up00 = self.db00(x_up00)
        
        # 48 x 256 x 256 -> 1 x 256 x 256
        # x_mask = self.final_conv(x_mask)  
        x_mask = self.final_out(x_up00)
        
        return x_mask

