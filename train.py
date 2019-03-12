import argparse
import numpy
import os
import shutil
import time
import torch

from torch.utils.data import DataLoader
from util.sampler import ImbalancedDatasetSampler
from util.average_meter import AverageMeter

from datasets import dataset_factory
from models import model_factory
from criterion import criterion_factory
from optimizer import optimizer_factory

from collections import OrderedDict


####################
# Input Parameters #
####################

parser = argparse.ArgumentParser()

def str2bool(value):
    return value.strip().lower() == 'true'

parser.add_argument('--model_arch',
                    default='resnet_fcn',
                    type=str,
                    help='Name of model architecture')

parser.add_argument('--dataset_name',
                    default='snapshot_serengeti',
                    type=str,
                    help='Name of training dataset.')

parser.add_argument('--dataset_dir',
                    default='/input',
                    help='Directory containing dataset.')

parser.add_argument('--models_dir',
                    default='/models',
                    help='Directory to output model checkpoints.')

parser.add_argument('--checkpoint_file',
                    default='',
                    type=str,
                    help='Path to latest checkpoint.')

parser.add_argument('--pretrained',
                    default=True,
                    type=str2bool,
                    help='Use pre-trained model.')

parser.add_argument('--params_to_train',
                    required=False,
                    nargs='+',
                    help='List of trainable params. ex: Including layer1 will'
                         'train all model params that contain layer1 in their'
                         'name. Use model.named_parameters() to see all names.')

parser.add_argument('--params_to_randomize',
                    default=[],
                    required=False,
                    nargs='+',
                    help='List of params to randomize in model. Parameters '
                         'that are not randomized will be restored from the '
                         'checkpoint.')

parser.add_argument('--train',
                    default=True,
                    type=str2bool,
                    help='Train the model')

parser.add_argument('--validate',
                    default=True,
                    type=str2bool,
                    help='Validate model on validation set')

parser.add_argument('--epochs',
                    default=30,
                    type=int,
                    help='Number of epochs to run.')

parser.add_argument('--start_epoch',
                    default=-1,
                    type=int,
                    help='Epoch to start training at (effects learning rate)')

parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='Mini batch size.')

parser.add_argument('--criterion',
                    default='CrossEntropyLoss',
                    type=str,
                    help='Loss function.')

parser.add_argument('--optimizer',
                    default='SGD',
                    type=str,
                    help='Network optimizer.')

parser.add_argument('--num_threads',
                    default=4,
                    type=int,
                    help='Number of data loading threads.')

parser.add_argument('--image_size',
                    default=448,
                    type=int,
                    help='Size of image (smaller dimension)')

parser.add_argument('--lr',
                    default=0.01,
                    type=float,
                    help='Initial learning rate.')

parser.add_argument('--lr_decay',
                    default=0.1,
                    type=float,
                    help='Amount to multiply lr every lr_decay_epochs.')

parser.add_argument('--lr_decay_epochs',
                    default=10,
                    type=int,
                    help='Learning rate decays every lr_decay_epochs.')

parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='Momentum.')

parser.add_argument('--weight_decay',
                    default=1e-4,
                    type=float,
                    help='Weight decay.')

parser.add_argument('--print_freq',
                    default=10,
                    type=int,
                    help='Print every n iterations.')


def main():

    ##############
    # Initialize #
    ##############

    print('** Initializing engine **')

    global args
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    for key, value in vars(args).items():
        print('{:20s}{:s}'.format(key, str(value)))

    if not os.path.isdir(args.models_dir):
        os.makedirs(args.models_dir)
    print()


    ###################
    # Create Datasets #
    ###################

    dataset = {}
    loader = {}

    # Create train loader
    if args.train:
        train_dir = os.path.join(args.dataset_dir, 'train')
        if os.path.isdir(train_dir):
            dataset['train'] = dataset_factory.get_dataset(args.dataset_name,
                'train', train_dir, args)
            loader['train'] = DataLoader(
                dataset['train'],
                batch_size=args.batch_size,
                num_workers=args.num_threads,
                pin_memory=args.cuda,
                sampler=ImbalancedDatasetSampler(dataset['train']))
        else:
            print("%s does not contain a 'train' directory. \
                  Training disabled." % train_dir)
            args.train = False

    # Create validation loader
    if args.validate:
        val_dir = os.path.join(args.dataset_dir, 'val')
        if os.path.isdir(val_dir):
            dataset['val'] = dataset_factory.get_dataset(args.dataset_name,
                'val', val_dir, args)
            loader['val'] = DataLoader(
                dataset['val'],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_threads,
                pin_memory=args.cuda)
        else:
            print("%s does not contain a 'val' directory. \
                  Validation disabled." % val_dir)
            args.validate = False

    if not args.train and not args.validate:
        raise ValueError('Training and validation cannot both be disabled.')


    ###############
    # Build Model #
    ###############

    print('** Building model **')

    data = dataset['train'] if args.train else dataset['val']
    model = model_factory.get_model(args.model_arch, data, args.pretrained)
    if args.cuda and torch.cuda.device_count() > 1:
        print("Loading model on %i cuda devices" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    if args.params_to_train:
        print('Collecting trainable parameters')
        params_to_train = []
        for param_name, param in model.named_parameters():
            param.requires_grad = False
            for word in args.params_to_train:
                if word in param_name:
                    param.requires_grad = True
                    params_to_train.append(param)
                    break
        print('Found %i trainable parameters' % len(params_to_train))
    else:
        params_to_train = model.parameters()
        print('Training all model parameters')

    # Define optimizer
    optimizer = optimizer_factory.get_optimizer(args.optimizer, params_to_train, args)

    # Define loss function
    criterion = criterion_factory.get_criterion(args.criterion, args.cuda)

    # Attempt to load model from checkpoint
    if args.checkpoint_file and os.path.isfile(args.checkpoint_file):
        print('Checkpoint found at: %s' % args.checkpoint_file)
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, args)
        print('Checkpoint successfully loaded')
    else:
        print('No checkpoint found')
        if not pretrained: print('Training from scratch')
        else: print('Parameters initialized with pytorch pretrained model')
        start_epoch = 0

    # Determine start epoch
    if args.start_epoch < 0:
        args.start_epoch = start_epoch

    # Load model to gpu/cpu
    if args.cuda:
        model.cuda()
    else:
        model.cpu()
    print()


    ############
    # Training #
    ############

    torch.backends.cudnn.benchmark = True

    if args.train:
        print('** Starting training **')
        for epoch in range(args.start_epoch, args.epochs):
            lr = adjust_learning_rate(optimizer, epoch)
            print('Epoch: %s, learning rate: %s' % (epoch, lr))

            # Train for one epoch
            train(model, loader['train'], criterion, optimizer, epoch)

            state = {
                'epoch': epoch + 1,
                'model_arch': args.model_arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }

            filename = 'checkpoint_%s-%i.pth.tar' % (args.model_arch, epoch + 1)
            torch.save(state, os.path.join(args.models_dir, filename))


    ##############
    # Validation #
    ##############

    if args.validate:
        print("** Evaluating model **")
        validate(model, loader['val'], criterion)


def train(model, dataloader, criterion, optimizer, epoch):
    """Train model on training set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(dataloader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Configure input data
        if args.cuda:
            input, target = input.cuda(async=True), target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Model inference
        output = model(input_var)

        # Compute output statistics
        loss = criterion(output, target_var)
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        losses.update(loss.data, input.size(0))

        # Compute gradient and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every args.print_freq
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                   epoch, i, len(dataloader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                   top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))


def validate(model, dataloader, criterion):
    """Validate model on validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(dataloader):
        # Configure input data
        if args.cuda:
            input, target = input.cuda(async=True), target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # Model inference
        output = model(input_var)

        # Compute output statistics
        loss = criterion(output, target_var)
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        losses.update(loss.data, input.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every args.print_freq
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                   i, len(dataloader), batch_time=batch_time,
                   loss=losses,
                   top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                   top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))

    print('* Prec@1 {top1}'.format(top1=numpy.asscalar(top1.avg.cpu().numpy())))


def load_checkpoint(model, optimizer, args):
    if args.cuda:
        checkpoint = torch.load(args.checkpoint_file)
        # try:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     for state in optimizer.state.values():
        #         for k, v in state.items():
        #             if isinstance(v, torch.Tensor):
        #                 state[k] = v.cuda()
        # except ValueError:
        #     print('Could not load optimizer state_dict')
        state_dict = retrieve_state_dict(checkpoint, args.cuda)
        model.load_state_dict(state_dict, strict=False)
    else:
        checkpoint = torch.load(args.checkpoint_file,
                                map_location=lambda storage, loc: storage)
        # try:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        # except ValueError:
        #     print('Could not load optimizer state_dict')
        state_dict = retrieve_state_dict(checkpoint, args.cuda)
        model.load_state_dict(state_dict, strict=False)
    start_epoch = checkpoint['epoch']

    return model, optimizer, start_epoch


def retrieve_state_dict(checkpoint, cuda):
    new_state_dict = OrderedDict()
    for param_name, param in checkpoint['state_dict'].items():
        # Remove 'module.' of dataparallel
        if not cuda:
            if param_name.startswith('module.'): param_name = param_name[7:]
            if param_name.startswith('0.'): param_name = param_name[2:]
        add_param = True
        for word in args.params_to_randomize:
            if word in param_name:
                add_param = False
                break
        if add_param: new_state_dict[param_name] = param
    return new_state_dict


def adjust_learning_rate(optimizer, epoch):
    """Adjusts learning rate every lr_decay_epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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


if __name__ == '__main__':
    main()
