import torch
from torch import nn

def BCELoss(cuda):
    criterion = nn.BCELoss()
    if cuda: criterion.cuda()
    return criterion

def BCEWithLogitsLoss(cuda):
    bce = nn.BCEWithLogitsLoss()
    if cuda: bce.cuda()
    def criterion(output, target):
        return bce(output.squeeze(), target.float())
    return criterion

def CrossEntropyLoss(cuda):
    criterion = nn.CrossEntropyLoss()
    if cuda: criterion.cuda()
    return criterion

criterion_map = {
    'BCELoss': BCELoss,
    'BCEWithLogitsLoss': BCEWithLogitsLoss,
    'CrossEntropyLoss': CrossEntropyLoss,
}

def get_criterion(args):
    return criterion_map[args.criterion](args.cuda)
