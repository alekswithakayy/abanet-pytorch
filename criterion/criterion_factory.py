import torch
from torch import nn

from .focal_loss import FocalLoss
from .ir_loss import InterPixelRelationLoss

def BCE(cuda):
    criterion = nn.BCELoss()
    if cuda: criterion.cuda()
    return criterion

def BCEWithLogits(cuda):
    bce = nn.BCEWithLogitsLoss()
    if cuda: bce.cuda()
    def criterion(output, target):
        return bce(output.squeeze(), target.float())
    return criterion

def CrossEntropy(cuda):
    criterion = nn.CrossEntropyLoss()
    if cuda: criterion.cuda()
    return criterion

def Focal(cuda):
    criterion = FocalLoss(alpha=0.25, gamma=2)
    if cuda: criterion.cuda()
    return criterion

def InterPixelRelation(cuda):
    criterion = InterPixelRelationLoss(pixel_radius=5)
    if cuda: criterion.cuda()
    return criterion

criterion_map = {
    'BCELoss': BCE,
    'BCEWithLogitsLoss': BCEWithLogits,
    'CrossEntropyLoss': CrossEntropy,
    'FocalLoss': Focal,
    'InterPixelRelationLoss': InterPixelRelation,
}

def get_criterion(args):
    return criterion_map[args.criterion](args.cuda)
