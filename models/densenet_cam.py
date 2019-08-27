from .densenet import densenet161

import torch.nn as nn
import torch.nn.functional as F

class DenseNetCAM(nn.Module):

    architecture = {'densenet161': densenet161}

    def __init__(self, args):
        super(DenseNetCAM, self).__init__()
        densenet = self.architecture[args.architecture](pretrained=args.pretrained)

        self.stage1 = nn.Sequential(densenet.features.conv0,
                                    densenet.features.norm0,
                                    densenet.features.relu0,
                                    densenet.features.pool0)
        self.stage2 = nn.Sequential(densenet.features.denseblock1,
                                    densenet.features.transition1)
        self.stage3 = nn.Sequential(densenet.features.denseblock2,
                                    densenet.features.transition2)
        self.stage4 = nn.Sequential(densenet.features.denseblock3,
                                    densenet.features.transition3)
        self.stage5 = nn.Sequential(densenet.features.denseblock4,
                                    densenet.features.norm5)

        self.features = nn.ModuleList(
            [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.classifier = nn.Conv2d(densenet.classifier.in_features,
            args.num_classes, 1, bias=False)
        self.newly_added = nn.ModuleList([self.classifier])


    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        logits = self.classifier(F.adaptive_avg_pool2d(F.relu(x5), (1,1)))
        logits = logits.squeeze()

        if self.training:
            return logits
        else:
            cams = F.conv2d(F.relu(x5), self.classifier.weight)
            cams = F.relu(cams)
            return logits, cams

    def trainable_parameters(self):
        return (list(self.features.parameters()), list(self.newly_added.parameters()))
