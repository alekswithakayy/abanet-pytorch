import re
import torch
import numpy

from collections import OrderedDict

def load_checkpoint(model, checkpoint_filepath, cuda, optimizer=None,
    params_to_randomize=None):
    if cuda:
        checkpoint = torch.load(checkpoint_filepath)
        # try:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.cuda()
        # except ValueError:
        #     print('Could not load optimizer state_dict')
        state_dict = _retrieve_state_dict(checkpoint, params_to_randomize, True)
        model.load_state_dict(state_dict, strict=True)
    else:
        checkpoint = torch.load(checkpoint_filepath,
                                map_location=lambda storage, loc: storage)
        # try:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        # except ValueError:
        #     print('Could not load optimizer state_dict')
        state_dict = _retrieve_state_dict(checkpoint, params_to_randomize, False)
        model.load_state_dict(state_dict, strict=True)
    start_epoch = checkpoint['epoch']

    return model, optimizer, start_epoch


def _retrieve_state_dict(checkpoint, params_to_randomize, cuda):
    param_regex = None
    if params_to_randomize:
        param_regex = re.compile(params_to_randomize)

    new_state_dict = OrderedDict()
    for param_name, param in checkpoint['state_dict'].items():
        # Remove 'module.' of dataparallel
        if torch.cuda.device_count() <= 1:
            if param_name.startswith('module.'): param_name = param_name[7:]
            if param_name.startswith('0.'): param_name = param_name[2:]
        if not param_regex or param_regex.search(param_name):
            new_state_dict[param_name] = param

    return new_state_dict


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
