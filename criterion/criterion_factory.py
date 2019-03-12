from torch import nn

def BCELoss(cuda):
    criterion = nn.BCELoss()
    if cuda: criterion.cuda()
    return criterion

def BCEWithLogitsLoss(cuda):
    criterion = nn.BCEWithLogitsLoss()
    if cuda: criterion.cuda()
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

def get_criterion(criterion_name, cuda):
    return criterion_map[criterion_name](cuda)
