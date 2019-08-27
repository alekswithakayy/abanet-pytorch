from .dilated_resnet import drn_d_54, drn_d_56, drn_d_105, drn_d_107
from .dilated_resnet import dilated_resnext56_32x4d, dilated_resnext107_32x8d

import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedResNetCAM(nn.Module):

    architectures = {'dilated_resnet54': drn_d_54,
                     'dilated_resnet56': drn_d_56,
                     'dilated_resnet105': drn_d_105,
                     'dilated_resnet107': drn_d_107,
                     'dilated_resnext56_32x4d': dilated_resnext56_32x4d,
                     'dilated_resnext107_32x8d': dilated_resnext107_32x8d}

    def __init__(self, args):
        super(DilatedResNetCAM, self).__init__()

        self.resnet = self.architectures[args.architecture](args.pretrained)

        self.stage1 = nn.Sequential(self.resnet.layer0,
                                    self.resnet.layer1,
                                    self.resnet.layer2)
        self.stage2 = nn.Sequential(self.resnet.layer3)
        self.stage3 = nn.Sequential(self.resnet.layer4)
        self.stage4 = nn.Sequential(self.resnet.layer5)
        self.stage5 = nn.Sequential(self.resnet.layer6,
                                    self.resnet.layer7,
                                    self.resnet.layer8)

        # self.dil_conv1 = nn.Conv2d(2048, 2048, kernel_size=3, bias=False, dilation=1, padding=1)
        # self.dil_conv3 = nn.Conv2d(2048, 2048, kernel_size=3, bias=False, dilation=3, padding=3)
        # self.dil_conv6 = nn.Conv2d(2048, 2048, kernel_size=3, bias=False, dilation=6, padding=6)
        # self.dil_conv9 = nn.Conv2d(2048, 2048, kernel_size=3, bias=False, dilation=9, padding=9)

        self.classifier = nn.Conv2d(512, args.num_classes, kernel_size=1, bias=False)

        self.backbone = nn.ModuleList(
            [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.newly_added = nn.ModuleList([self.classifier])


    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        # d1 = F.relu(self.dil_conv1(x5))
        # d3 = F.relu(self.dil_conv3(x5))
        # d6 = F.relu(self.dil_conv6(x5))
        # d9 = F.relu(self.dil_conv9(x5))
        # dil_out = [d1, d3, d6, d9]
        #
        # p = []
        # for d in dil_out:
        #     p.append(F.adaptive_avg_pool2d(d, (1,1)))
        # pool_out = torch.cat(p, 2)
        # logits = torch.sum(self.classifier(pool_out), 2).squeeze()
        #
        # if self.training:
        #     return logits
        # else:
        #     cams = []
        #     for d in dil_out:
        #         cams.append(F.relu(F.conv2d(d, self.classifier.weight)))
        #     cams = torch.stack(cams)
        #     cams = cams[0] + torch.mean(cams[1:], 0)
        #     return logits, cams

        logits = self.classifier(F.adaptive_avg_pool2d(x5, (1,1)))
        logits = logits.squeeze()

        if self.training:
            return logits
        else:
            cams = F.conv2d(x5, self.classifier.weight)
            cams = F.relu(cams)
            return logits, cams

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
