"""Contains a factory for building various models."""

import torch

from .resnet_fcn import ResNetFCN
from .resnet_ps import ResNetPS
from .densenet_fcn import DenseNetFCN
from .densenet_ps import DenseNetPS

models_map = {'resnet_fcn': ResNetFCN,
              'resnet_ps': ResNetPS,
              'densenet_fcn': DenseNetFCN,
              'densenet_ps': DenseNetPS,
             }

def get_model(model_name, dataset, pretrained):
    if model_name not in models_map:
        raise ValueError('Name of network unknown %s' % model_name)

    model = models_map[model_name](dataset, pretrained)

    return model
