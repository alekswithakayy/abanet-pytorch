"""Contains a factory for building various models."""

import torch

from .resnet_fcn import ResNetFCN
from .densenet_fcn import DenseNetFCN
from prm.peak_response_mapping import PeakResponseMapping

models_map = {'resnet_fcn': ResNetFCN,
              'densenet_fcn': DenseNetFCN,
             }

def get_model(name, num_classes, pretrained, prm, peak_std):
    if name not in models_map:
        raise ValueError('Name of network unknown %s' % name)

    model = models_map[name](num_classes, pretrained)

    if prm:
        model = PeakResponseMapping(model, enable_peak_backprop=True,
            win_size=3, sub_pixel_locating_factor=1)

    return model
