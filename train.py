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
from util import load_checkpoint, accuracy, AverageMeter, MovingAverageMeter
from apex import amp

def run(train_args, dataset_args, model_args):

    ##############
    # Initialize #
    ##############

    print('** Initializing training engine **')

    train_args.cuda = torch.cuda.is_available()

    for key, value in vars(train_args).items():
        print('{:20s}{:s}'.format(key, str(value)))
    print()

    if not os.path.isdir(train_args.models_dir):
        os.makedirs(train_args.models_dir)


    ##################
    # Create Dataset #
    ##################

    print('** Loading dataset **')

    for key, value in vars(dataset_args).items():
        print('{:20s}{:s}'.format(key, str(value)))
    print()

    dataset = dataset_factory.get_dataset('train', dataset_args)
    dataloader = DataLoader(dataset, batch_size=dataset_args.batch_size,
        num_workers=dataset_args.num_threads, pin_memory=train_args.cuda,
        sampler=ImbalancedDatasetSampler(dataset, dataset_args))

    print('Found %s samples in training set' % len(dataset))
    print()


    ###############
    # Build Model #
    ###############

    print('** Building model **')

    for key, value in vars(model_args).items():
        print('{:20s}{:s}'.format(key, str(value)))
    print()

    # Define model
    model = model_factory.get_model(model_args, train_args.cuda)

    # Collect and configure trainable params
    if train_args.params_to_train:
        print('Collecting trainable parameters')
        param_regex = re.compile(train_args.params_to_train)
        train_args.params_to_train = []
        for param_name, param in model.named_parameters():
            if param_regex.search(param_name):
                param.requires_grad = True
                train_args.params_to_train.append(param)
            else: param.requires_grad = False
        print('Found %i trainable parameters' % len(train_args.params_to_train))
    else:
        train_args.params_to_train = model.parameters()
        print('Training all model parameters')

    # Define optimizer
    optimizer = optimizer_factory.get_optimizer(train_args)

    # Define loss function
    criterion = criterion_factory.get_criterion(train_args)

    # Attempt to load model from checkpoint
    if model_args.checkpoint and os.path.isfile(model_args.checkpoint):
        print('Checkpoint found at: %s' % model_args.checkpoint)
        model, optimizer, start_epoch = load_checkpoint(model,
            model_args.checkpoint, train_args.cuda, optimizer=optimizer,
            params_to_randomize=train_args.params_to_randomize)
        print('Checkpoint successfully loaded')
    else:
        print('No checkpoint found at: %s' % model_args.checkpoint)
        if not model_args.pretrained:
            print('Training from scratch')
        else:
            print('Parameters initialized with pytorch pretrained model')
        start_epoch = 0
    print()

    # Determine start epoch and current iteration
    if train_args.start_epoch < 0:
        train_args.start_epoch = start_epoch
    train_args.cur_iter = train_args.start_epoch * len(dataloader)
    # Remove lr_decay events that have occured
    while train_args.lr_decay_iters:
        if train_args.lr_decay_iters[0] <= train_args.cur_iter:
            train_args.lr_decay_iters.pop(0)
            continue
        break

    # cuDNN looks for the optimal set of algorithms for network configuration
    # Benchmark mode is good whenever input sizes do not vary
    torch.backends.cudnn.benchmark = True

    # Initialize Nvidia/apex for mixed prec training
    print('Preparing model for mixed precision training')
    model, optimizer = amp.initialize(model, optimizer,
        opt_level=train_args.mixed_prec_level)
    print()


    #########
    # Train #
    #########

    print('** Beginning training **')

    model.train()

    for epoch in range(train_args.start_epoch, train_args.epochs):
        print('Epoch: %s' % epoch)

        start = time.time()

        # Train for one epoch
        train(model, dataloader, criterion, optimizer, epoch, train_args)

        epoch_time = (time.time() - start) / 60 / 60
        print('Epoch time: %.2f hours\n' %  epoch_time)

        state = {
            'epoch': epoch + 1,
            'architecture': model_args.architecture,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }

        filename = 'checkpoint_%s-%i.pth.tar' \
            % (model_args.architecture, epoch + 1)
        torch.save(state, os.path.join(train_args.models_dir, filename))


def train(model, dataloader, criterion, optimizer, epoch, train_args):
    """Train model on training set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = MovingAverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(dataloader):
        train_args.cur_iter += 1

        # Measure data loading time
        data_time.update(time.time() - end)

        # Configure input data
        if train_args.cuda:
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
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # Update learning rate
        if train_args.lr_decay_iters:
            if train_args.lr_decay_iters[0] < train_args.cur_iter:
                train_args.lr_decay_iters.pop(0)
                train_args.lr = train_args.lr * train_args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = train_args.lr
                print('Learning rate adjusted to: %s' % train_args.lr)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every print_freq
        if i % train_args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1_val} ({top1_avg})'.format(
                   epoch, i, len(dataloader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   top1_val=numpy.asscalar(top1.val.cpu().numpy()),
                   top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))
