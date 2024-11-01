import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class AlphaPoseModel(nn.Module):
    def __init__(self, num_keypoints):
        super(AlphaPoseModel, self).__init__()
        # Use the weights parameter instead of pretrained
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the last fc layer and avgpool
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])
        self.final_layer = nn.Conv2d(in_channels=256, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=2048 if i==0 else num_filters[i-1],
                                             out_channels=planes,
                                             kernel_size=kernel,
                                             stride=2,
                                             padding=1,
                                             output_padding=0,
                                             bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x
