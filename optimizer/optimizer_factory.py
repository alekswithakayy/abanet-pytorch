from torch import optim

def SGD(params_to_train, args):
    return optim.SGD(params_to_train, args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

def Adam(params_to_train, args):
    return optim.Adam(params_to_train, args.lr, weight_decay=args.weight_decay)

optimizer_map = {
    'SGD': SGD,
    'Adam': Adam,
}

def get_optimizer(optimizer_name, params_to_train, args):
    return optimizer_map[optimizer_name](params_to_train, args)
