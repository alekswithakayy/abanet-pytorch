import numpy
import os
import shutil
import time
import torch
import re

from torch.utils.data import DataLoader
from datasets import dataset_factory
from models import model_factory
from criterion import criterion_factory
from optimizer import optimizer_factory
from util.sampler import ImbalancedDatasetSampler
from util import load_checkpoint, accuracy, AverageMeter


def run(args):

    ##############
    # Initialize #
    ##############

    print('** Initializing testing engine **')

    for key, value in vars(args).items():
        print('{:20s}{:s}'.format(key, str(value)))
    print()


    ##################
    # Create Dataset #
    ##################

    print('** Loading dataset **')

    dataset = dataset_factory.get_dataset('test', args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_threads,
        pin_memory=args.cuda)

    print('Found %s samples in testing set' % len(dataset))
    print()


    ###############
    # Build Model #
    ###############

    print('** Building model **')

    for key, value in vars(args).items():
        print('{:20s}{:s}'.format(key, str(value)))
    print()

    # Define model
    model = model_factory.get_model(args)

    # Define loss function
    criterion = criterion_factory.get_criterion(args)

    # Attempt to load model from checkpoint
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print('Checkpoint found at: %s' % args.checkpoint)
        model, _, _ = load_checkpoint(model, args.checkpoint, args.cuda)
        print('Checkpoint successfully loaded')
    else:
        print('No checkpoint found at: %s' % args.checkpoint)
        print('Halting testing')
        return
    print()

    # cuDNN looks for the optimal set of algorithms for network configuration
    # Benchmark mode is good whenever input sizes do not vary
    torch.backends.cudnn.benchmark = True


    ########
    # Test #
    ########

    print("** Beginning testing **")

    model.eval()

    test(model, dataloader, criterion, args.cuda, args.test_print_freq)


def test(model, dataloader, criterion, cuda, print_freq):
    """Test model on test set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(dataloader):
        # Configure input data
        if cuda:
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

        # Info log every print_freq
        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                   i, len(dataloader), batch_time=batch_time,
                   loss=losses,
                   top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                   top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))

    print('* Prec@1 {top1}'.format(top1=numpy.asscalar(top1.avg.cpu().numpy())))
