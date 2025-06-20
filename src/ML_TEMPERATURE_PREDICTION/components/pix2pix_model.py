# Update file pix2pix_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class UNetDown(nn.Module):
    """
    UNet downsampling module
    """
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2))
        
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """
    UNet upsampling module
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip_input):
        x = self.model(x)
        # Add padding if necessary to match dimensions
        if x.shape != skip_input.shape:
            # Calculate padding needed
            diffY = skip_input.size()[2] - x.size()[2]
            diffX = skip_input.size()[3] - x.size()[3]
            
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat((x, skip_input), 1)
        
        return x

class Generator(nn.Module):
    """
    Enhanced Generator architecture (modified UNet) with configurable filters
    """
    def __init__(self, in_channels=3, out_channels=1, filters=64):
        super(Generator, self).__init__()
        
        # Initial layer
        self.down1 = UNetDown(in_channels, filters, normalize=False)
        # Downsampling layers
        self.down2 = UNetDown(filters, filters*2)
        self.down3 = UNetDown(filters*2, filters*4)
        self.down4 = UNetDown(filters*4, filters*8, dropout=0.5)
        self.down5 = UNetDown(filters*8, filters*8, dropout=0.5)
        self.down6 = UNetDown(filters*8, filters*8, dropout=0.5)
        self.down7 = UNetDown(filters*8, filters*8, dropout=0.5)
        self.down8 = UNetDown(filters*8, filters*8, normalize=False, dropout=0.5)
        
        # Upsampling layers
        self.up1 = UNetUp(filters*8, filters*8, dropout=0.5)
        self.up2 = UNetUp(filters*16, filters*8, dropout=0.5)
        self.up3 = UNetUp(filters*16, filters*8, dropout=0.5)
        self.up4 = UNetUp(filters*16, filters*8, dropout=0.5)
        self.up5 = UNetUp(filters*16, filters*4)
        self.up6 = UNetUp(filters*8, filters*2)
        self.up7 = UNetUp(filters*4, filters)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(filters*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Upsampling
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)

class Discriminator(nn.Module):
    """
    Enhanced Discriminator architecture (PatchGAN) with configurable filters
    """
    def __init__(self, in_channels=3, out_channels=1, filters=64):
        super(Discriminator, self).__init__()
        
        # Input: A + B (concatenated)
        # For RGB to Thermal: A=RGB (3 channels), B=Thermal (1 channel) => total = 4
        # For Thermal to RGB: A=Thermal (1 channel), B=RGB (3 channels) => total = 4
        total_input_channels = in_channels + out_channels
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # Input layer
            *discriminator_block(total_input_channels, filters, normalize=False),
            
            # Middle layers - using configurable filters
            *discriminator_block(filters, filters*2),
            *discriminator_block(filters*2, filters*4),
            *discriminator_block(filters*4, filters*8),
            
            # Output layer
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(filters*8, 1, kernel_size=4, padding=1, bias=False)
        )
    
    def forward(self, img_A, img_B):
        # Concatenate image and condition along channel dimension
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)