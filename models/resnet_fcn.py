import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetFCN(nn.Module):

    def __init__(self, num_classes, pretrained):
        super(ResNetFCN, self).__init__()

        model = models.resnet50(pretrained)

        # Remove last two layers (avg_pool and fc) of ResNet
        self.features = nn.Sequential(*list(model.children())[:-2])

        # Create new classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
