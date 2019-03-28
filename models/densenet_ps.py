"""Densenet with Peak Stimulation"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from prm import PeakStimulation

class DenseNetPS(nn.Module):

    def __init__(self, args):
        super(DenseNetPS, self).__init__()
        self.return_peaks = args.return_peaks
        # If only two classes, configure
        # for binary cross entropy
        if args.num_classes == 2:
            args.num_classes = 1

        # Retrieve pretrained densenet
        model = models.densenet161(pretrained=args.pretrained)
        self.features = model.features

        # Create new classification layer
        n_features = model.classifier.in_features
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
