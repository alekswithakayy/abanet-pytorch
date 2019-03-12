"""Densenet with Peak Stimulation"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from prm.peak_stimulation import PeakStimulation

class DenseNetPS(nn.Module):

    def __init__(self, dataset, pretrained):
        super(DenseNetFCN, self).__init__()
        
        self.n_classes = len(dataset.classes)
        # If only two classes, configure
        # for binary cross entropy
        if self.n_classes == 2:
            self.n_classes = 1

        # Retrieve pretrained densenet
        model = models.densenet161(pretrained=pretrained)
        self.features = model.features

        # Create new classification layer
        n_features = model.classifier.in_features
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_features, self.n_classes, kernel_size=1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        _, x = PeakStimulation.apply(x, 3, self.training)
        return x
