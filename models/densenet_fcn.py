import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DenseNetFCN(nn.Module):

    def __init__(self, n_classes, pretrained):
        super(DenseNetFCN, self).__init__()

        model = models.densenet161(pretrained=pretrained)

        self.features = model.features

        # Create new classification layer
        n_features = model.classifier.in_features
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(n_features, n_classes, kernel_size=1))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x).squeeze()
        return x
