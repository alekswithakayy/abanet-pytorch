"""Resnet with global weighted average pooling classifier"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetGWAP(nn.Module):

    def __init__(self, args):
        super(ResNetGWAP, self).__init__()
        # Retrieve pretrained resnet
        model = models.resnet101(pretrained=args.pretrained)

        # Remove last two layers (avg_pool and fc) of ResNet
        self.features = nn.Sequential(*list(model.children())[:-2])

        # Create new classification layer
        n_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_features, args.num_classes, kernel_size=1))

    def forward(self, x):
        x = self.features(x)
        activation_maps = self.classifier(x)
        logits = global_weighted_avg_pool2D(activation_maps)
        if not self.training:
            return logits, activation_maps
        else:
            return logits


def global_weighted_avg_pool2D(x):
    b, c, h, w = x.size()
    O_c = F.softmax(x, dim=1)
    M_c = O_c * torch.sigmoid(x)
    alpha_c = F.softmax(M_c.view(b, c, h*w), dim=2)
    x = alpha_c * x.view(b, c, h*w)
    x = torch.sum(x, dim=2).squeeze()
    return x
