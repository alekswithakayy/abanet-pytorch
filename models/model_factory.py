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

def get_model(args, cuda):
    if args.architecture not in models_map:
        raise ValueError('Name of network unknown %s' % args.architecture)

    model = models_map[args.architecture](args)

    # Load model to gpu/cpu
    if cuda:
        if torch.cuda.device_count() > 1:
            print("Loading model on %i cuda devices" % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
        model.cuda()
    else:
        model.cpu()

    return model
