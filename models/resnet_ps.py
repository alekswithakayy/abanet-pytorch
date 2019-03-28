"""Resnet with Peak Stimulation"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from prm.peak_stimulation import PeakStimulation

class ResNetPS(nn.Module):

    def __init__(self, args):
        super(ResNetPS, self).__init__()
        self.return_peaks = args.return_peaks
        # If only two classes, configure
        # for binary cross entropy
        if args.num_classes == 2:
            args.num_classes = 1

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
        # Get class response maps
        crms = self.classifier(x)
        # Stimulate peak formation
        peaks, logits = PeakStimulation.apply(crms, 3, self.training)
        if self.return_peaks:
            return logits, crms, peaks
        else:
            return logits
