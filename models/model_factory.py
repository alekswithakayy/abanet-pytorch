"""Contains a factory for building various models."""

import torch

from .resnet_fcn import ResNetFCN
from .densenet_fcn import DenseNetFCN

models_map = {'resnet_fcn': ResNetFCN,
              'densenet_fcn': DenseNetFCN,
             }

def get_model(name, n_classes, pretrained):
    if name not in models_map:
        raise ValueError('Name of network unknown %s' % name)

    model = models_map[name](n_classes, pretrained)

    return model
