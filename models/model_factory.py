"""Contains a factory for building various models."""

import torch

from .resnet_fcn import ResNetFCN
from prm.peak_response_mapping import PeakResponseMapping

models_map = {'resnet_fcn': ResNetFCN,
             }

def get_model(name, num_classes, pretrained, prm):
    if name not in models_map:
        raise ValueError('Name of network unknown %s' % name)

    model = models_map[name](num_classes, pretrained)

    if prm:
        model = peak_response_mapping(model)

    return model

def peak_response_mapping(backbone, enable_peak_stimulation=True,
    enable_peak_backprop=True, win_size=5, sub_pixel_locating_factor=1,
    filter_type='mean'):
    """Peak Response Mapping.
    """

    return PeakResponseMapping(
        backbone,
        enable_peak_stimulation=enable_peak_stimulation,
        enable_peak_backprop=enable_peak_backprop,
        win_size=win_size,
        sub_pixel_locating_factor=sub_pixel_locating_factor,
        filter_type=filter_type)
