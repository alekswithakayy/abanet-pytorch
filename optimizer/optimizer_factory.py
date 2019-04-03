from torch import optim
from .yellowfin import YFOptimizer

def SGD(args):
    return optim.SGD(args.params_to_train, args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

def Adam(args):
    return optim.Adam(args.params_to_train, args.lr,
        weight_decay=args.weight_decay)

def YellowFin(args):
    # default from paper: lr=1.0, mu=0.0, weight_decay=5e-4
    # lr_decay and lr_decay_epochs should be 0
    return YFOptimizer(args.params_to_train, lr=args.lr, mu=args.momentum,
        weight_decay=args.weight_decay)

optimizer_map = {
    'SGD': SGD,
    'Adam': Adam,
    'YellowFin': YellowFin,
}

def get_optimizer(args):
    return optimizer_map[args.optimizer](args)
