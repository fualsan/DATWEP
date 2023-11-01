import torch
import torch.nn as nn


class DownscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu_inplace):
        super().__init__()
        
        self.downscale_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.MaxPool2d(kernel_size=(2,2))
        )

    def forward(self, x):
        return self.downscale_block(x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu_inplace):
        super().__init__()
        
        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=relu_inplace),
        )
        
    def forward(self, x):
        return self.bottleneck_block(x)


class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu_inplace):
        super().__init__() 
        
        self.upscale_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=relu_inplace),
        )
        
    def forward(self, x):
        return self.upscale_block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, features=32, relu_inplace=True):
        super().__init__()
        self.n_classes = n_classes 

        self.down_3_32 = DownscaleBlock(in_channels, features, relu_inplace)

        self.down_32_64 = DownscaleBlock(features, features*2, relu_inplace)
        
        self.down_64_128 = DownscaleBlock(features*2, features*4, relu_inplace)
        
        self.bottleneck = BottleneckBlock(features*4, features*8, relu_inplace)

        self.up_384_128 = UpscaleBlock(features*8 + features*4, features*4, relu_inplace)

        self.up_192_64 =  UpscaleBlock(features*4 + features*2, features*4, relu_inplace)

        self.up_92_32 =  UpscaleBlock(features*4 + features, features*2, relu_inplace)
        
        self.final = nn.Conv2d(in_channels=features*2, out_channels=n_classes, kernel_size=(3,3), stride=1, padding=1)

        
    def forward(self, x):
        # Downsample
        x_res1 = self.down_3_32(x)
        x_res2 = self.down_32_64(x_res1)
        x_res3 = self.down_64_128(x_res2)
        
        bottleneck = self.bottleneck(x_res3)
        
        
        # Upsampling
        up1 = torch.cat((x_res3, bottleneck), dim=1)
        up1 = self.up_384_128(up1)
        
        up2 = torch.cat((x_res2, up1), dim=1)
        up2 = self.up_192_64(up2)
        
        up3 = torch.cat((x_res1, up2), dim=1)
        up3 = self.up_92_32(up3)
        
        # Final class convertion
        final_output = self.final(up3)
        
        return final_output, (x_res1, x_res2, x_res3)

