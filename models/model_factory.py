"""Contains a factory for building various models."""

import torch
from .resnet_cam import ResNetCAM
from .densenet_cam import DenseNetCAM
from .dilated_resnet_cam import DilatedResNetCAM

models_map = {'resnet50': ResNetCAM,
              'resnet101': ResNetCAM,
              'resnext50_32x4d': ResNetCAM,
              'resnext101_32x8d': ResNetCAM,
              'densenet161': DenseNetCAM,
              'dilated_resnet54': DilatedResNetCAM,
              'dilated_resnet56': DilatedResNetCAM,
              'dilated_resnet105': DilatedResNetCAM,
              'dilated_resnet107': DilatedResNetCAM,
              'dilated_resnext56_32x4d': DilatedResNetCAM,
              'dilated_resnext107_32x8d': DilatedResNetCAM
             }

def get_model(args):
    if args.architecture not in models_map:
        raise ValueError('Name of network unknown %s' % args.architecture)
    model = models_map[args.architecture](args)
    return model
