import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import random
from torch.utils.data import Subset

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PoseNet(nn.Module):
    def __init__(self, num_keypoints):
        super(PoseNet, self).__init__()
        # Load pretrained ResNet50 as backbone
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the last two layers (avgpool and fc)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Refinement stages
        self.stage1 = nn.Sequential(
            ConvBlock(2048, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )
        
        self.stage2 = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )
        
        self.stage3 = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64)
        )
        
        # Upsampling layers
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        
        # Final layers for keypoint prediction
        self.final_stage = nn.Sequential(
            ConvBlock(64, 32),
            nn.Conv2d(32, num_keypoints, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone feature extraction
        x = self.backbone(x)
        
        # Refinement stages with upsampling
        x = self.stage1(x)
        x = self.deconv1(x)
        
        x = self.stage2(x)
        x = self.deconv2(x)
        
        x = self.stage3(x)
        x = self.deconv3(x)
        
        # Final prediction
        x = self.final_stage(x)
        
        return x