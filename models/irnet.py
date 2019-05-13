"""IRNet from 'Weakly Supervised Learning of Instance Segmentation with
   Inter-pixel Relations'"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import matplotlib.pyplot as plt

class IRNet(nn.Module):

    def __init__(self, args):
        super(IRNet, self).__init__()

        # Retrieve pretrained densenet
        model = models.densenet161(pretrained=args.pretrained)

        # Create references to network blocks
        self.features = nn.Sequential(
            model.features[:4],
            model.features[4:6],
            model.features[6:8],
            model.features[8:10],
            model.features[10],
            model.features[11])

        # Displacement field predictor sub-network
        self.disp_field_base = nn.ModuleList([
            BatchNormReLUConv2D(96, 64),
            BatchNormReLUConv2D(192, 128),
            BatchNormReLUConv2D(384, 256),
            BatchNormReLUConv2D(1056, 256),
            BatchNormReLUConv2D(2208, 256),
            BatchNormReLUConv2D(768, 256)])
        self.disp_field_head = nn.Sequential(
            BatchNormReLUConv2D(448, 256),
            BatchNormReLUConv2D(256, 256),
            BatchNormReLUConv2D(256, 2))

        # Boundary detection sub-network
        self.boundary_detect_base = nn.ModuleList([
            BatchNormReLUConv2D(96, 64),
            BatchNormReLUConv2D(192, 128),
            BatchNormReLUConv2D(384, 256),
            BatchNormReLUConv2D(1056, 256),
            BatchNormReLUConv2D(2208, 256)])
        self.boundary_detect_head = nn.Sequential(
            BatchNormReLUConv2D(960, 256))

        # Classification layer
        n_features = model.classifier.in_features
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_features, args.num_classes, kernel_size=1))


    def forward(self, x):
        _,_,h,w = x.shape
        disp_field_output = []
        boundary_detect_output = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i == 0:
                x_ = F.interpolate(x, size=(h//8,w//8), mode='bilinear')
                d_out = self.disp_field_base[i](x_)
                disp_field_output.append(d_out)
                b_out = self.boundary_detect_base[i](x_)
                boundary_detect_output.append(b_out)
            elif i == 1:
                d_out = self.disp_field_base[i](x)
                disp_field_output.append(d_out)
                b_out = self.boundary_detect_base[i](x)
                boundary_detect_output.append(b_out)
            elif i < 5:
                d_out = self.disp_field_base[i](x)
                d_out = F.interpolate(d_out, size=(h//16,w//16), mode='bilinear')
                disp_field_output.append(d_out)
                b_out = self.boundary_detect_base[i](x)
                b_out = F.interpolate(b_out, size=(h//8,w//8), mode='bilinear')
                boundary_detect_output.append(b_out)

        out = self.disp_field_base[5](torch.cat(disp_field_output[2:], dim=1))
        out = F.interpolate(out, size=(h//8,w//8), mode='bilinear')
        disp_field_output = disp_field_output[:2] + [out]
        disp_field = self.disp_field_head(torch.cat(disp_field_output, dim=1))

        boundary_detect = self.boundary_detect_head(
            torch.cat(boundary_detect_output, dim=1))

        if not self.training:
            activation_maps = self.classifier(x)
            logits = global_weighted_avg_pool2D(activation_maps)
            return logits, activation_maps, disp_field, boundary_detect
        else:
            return disp_field, boundary_detect


def global_weighted_avg_pool2D(x):
    b, c, h, w = x.size()
    O_c = F.softmax(x, dim=1)
    M_c = O_c * torch.sigmoid(x)
    alpha_c = F.softmax(M_c.view(b, c, h*w), dim=2)
    x = alpha_c * x.view(b, c, h*w)
    x = torch.sum(x, dim=2).squeeze()
    return x


class BatchNormReLUConv2D(nn.Sequential):
    def __init__(self, n_input_feats, n_output_feats, kernel_size=1):
        super(BatchNormReLUConv2D, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(n_input_feats))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(n_input_feats, n_output_feats,
            kernel_size=kernel_size, stride=1, bias=False))
