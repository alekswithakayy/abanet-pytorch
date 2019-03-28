from torch import optim

def SGD(args):
    return optim.SGD(args.params_to_train, args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

def Adam(args):
    return optim.Adam(args.params_to_train, args.lr,
        weight_decay=args.weight_decay)

optimizer_map = {
    'SGD': SGD,
    'Adam': Adam,
}

def get_optimizer(args):
    return optimizer_map[args.optimizer](args)
