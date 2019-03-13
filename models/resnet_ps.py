"""Resnet with Peak Stimulation"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetPS(nn.Module):

    def __init__(self, dataset, pretrained):
        super(ResNetPS, self).__init__()

        self.n_classes = len(dataset.classes)
        # If only two classes, configure
        # for binary cross entropy
        if self.n_classes == 2:
            self.n_classes = 1

        # Retrieve pretrained resnet
        model = models.resnet101(pretrained=pretrained)

        # Remove last two layers (avg_pool and fc) of ResNet
        self.features = nn.Sequential(*list(model.children())[:-2])

        # Create new classification layer
        n_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_features, self.n_classes, kernel_size=1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        _, x = PeakStimulation.apply(x, 3, self.training)
        return x
