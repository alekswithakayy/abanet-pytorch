import argparse
import numpy
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.optim import SGD
from torch.utils.data import DataLoader

from util import sampler
from datasets import dataset_factory
from models import model_factory


####################
# Input Parameters #
####################

parser = argparse.ArgumentParser()

def str2bool(value):
    return value.strip().lower() == 'true'

parser.add_argument('--dataset_dir',
                    default='/input',
                    help='Directory containing dataset.')

parser.add_argument('--dataset_name',
                    default='snapshot_serengeti',
                    type=str,
                    help='Name of training dataset.')

parser.add_argument('--model_dir',
                    default='/models',
                    help='Directory to output model checkpoints.')

parser.add_argument('--model_arch',
                    default='resnet_fcn',
                    type=str,
                    help='Name of model architecture')

parser.add_argument('--num_threads',
                    default=4,
                    type=int,
                    help='Number of data loading threads.')

parser.add_argument('--epochs',
                    default=30,
                    type=int,
                    help='Number of epochs to run.')

parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='Mini batch size.')

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

parser.add_argument('--prm',
                    default=True,
                    type=str2bool,
                    help='Enable peak response mapping.')

parser.add_argument('--print_freq',
                    default=10,
                    type=int,
                    help='Print every n iterations.')

parser.add_argument('--checkpoint',
                    default='',
                    type=str,
                    help='Path to latest checkpoint.')

parser.add_argument('--train',
                    default=True,
                    type=str2bool,
                    help='Train the model')

parser.add_argument('--evaluate',
                    default=True,
                    type=str2bool,
                    help='Evaluate model on validation set')

parser.add_argument('--trainable_params',
                    required=False,
                    nargs='+',
                    help='List of trainable params. ex: Including layer1 will'
                         'train all model params that contain layer1 in their'
                         'name. Use model.named_parameters() to see all names.')

parser.add_argument('--pretrained',
                    default=True,
                    type=str2bool,
                    help='Use pre-trained model.')


def main():

    ##############
    # Initialize #
    ##############

    global args, best_prec1, cuda, labels

    args = parser.parse_args()

    if not os.path.isdir(args.model_dir): os.makedirs(args.model_dir)

    # Check if cuda is available
    cuda = torch.cuda.is_available()
    print("Using cuda: %s" % cuda)


    ###################
    # Create Datasets #
    ###################

    # Create train loader
    train_dir = os.path.join(args.dataset_dir, 'train')
    train_dataset = dataset_factory.get_dataset(args.dataset_name, 'train',
                                                train_dir)
    train_sampler = sampler.ImbalancedDatasetSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
        num_workers=args.num_threads, pin_memory=cuda, sampler=train_sampler)

    # Create validation loader
    val_dir = os.path.join(args.dataset_dir, 'val')
    val_dataset = dataset_factory.get_dataset(args.dataset_name, 'val', val_dir)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_threads, pin_memory=cuda)


    ###############
    # Build Model #
    ###############

    num_classes = len(train_dataset.classes)
    model = model_factory.get_model(args.model_arch, num_classes,
                                    args.pretrained, args.prm)

    if args.trainable_params:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze and collect all trainable parameters
        trainable_params = []
        trainable_param_count = 0
        for name, param in model.named_parameters():
            print('Searching for parameter names containing: %s' % trainable_params)
            for word in args.trainable_params:
                if word in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                    trainable_param_count += 1
                    break
        print('Training %i model parameters' % trainable_param_count)
    else:
        trainable_params = model.parameters()
        print('Training all model parameters')

    if cuda and torch.cuda.device_count() > 1:
        print("Loading model on %i cuda devices" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # Define loss function
    criterion = nn.CrossEntropyLoss()
    if cuda: criterion.cuda()

    # Define optimizer
    optimizer = SGD(trainable_params, args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)

    # Load model from checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("Checkpoint found at: %s" % args.checkpoint)
            if cuda:
                checkpoint = torch.load(args.checkpoint
                optimizer.load_state_dict(checkpoint['optimizer'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.cuda()
            else:
                checkpoint = torch.load(args.checkpoint,
                                        map_location=lambda storage, loc: 'cpu')
                optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint at epoch: %i" % args.start_epoch)
        else:
            print("No checkpoint found at: %s" % args.checkpoint)
    else:
        args.start_epoch = 0
        best_prec1 = torch.FloatTensor([0])

    # Load model on GPU or CPU
    if cuda: model.cuda()
    else: model.cpu()


    ############
    # Training #
    ############

    cudnn.benchmark = True

    if args.train:
        print("Starting training...")
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # Train for one epoch
            train(model, train_loader, criterion, optimizer, epoch)

            # Evaluate on validation set
            prec1 = validate(model, val_loader, criterion)

            # Remember best prec1 and save checkpoint
            if cuda: prec1 = prec1.cpu()

            is_best = bool(prec1.numpy() > best_prec1.numpy())
            best_prec1 = torch.FloatTensor(max(prec1.numpy(), best_prec1.numpy()))
            save_checkpoint({
                'epoch': epoch + 1,
                'model_arch': args.model_arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


    ##############
    # Evaluation #
    ##############

    if args.evaluate:
        print("Evaluating model...")
        validate(model, val_loader, criterion)
        return


def train(model, dataloader, criterion, optimizer, epoch):
    """Train model on training set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if cuda:
            input, target = input.cuda(async=True), target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)

        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o, target_var) for o in output))
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.data[0], input.size(0)*len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.data, input.size(0))

        # Compute gradient and do SGD step
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
    """Evaluate model on validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(dataloader):
        if cuda:
            input, target = input.cuda(async=True), target.cuda(async=True)

        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        output = model(input_var)

        # For nets that have multiple outputs such as Inception
        if isinstance(output, tuple):
            loss = sum((criterion(o,target_var) for o in output))
            # print (output)
            for o in output:
                prec1 = accuracy(o.data, target, topk=(1,))
                top1.update(prec1[0], input.size(0))
            losses.update(loss.data[0], input.size(0)*len(output))
        else:
            loss = criterion(output, target_var)
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))
            losses.update(loss.data, input.size(0))

        # measure elapsed time
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

    print(' * Prec@1 {top1}'
          .format(top1=numpy.asscalar(top1.avg.cpu().numpy())))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.model_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.model_dir, filename),
                        os.path.join(args.model_dir,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Adjusts learning rate every lr_decay_epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
