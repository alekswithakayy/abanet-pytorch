import numpy
import os
import time
import torch

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import model_factory
from util.sampler import ImbalancedDatasetSampler
from util import accuracy, AverageMeter, MovingAverageMeter
from apex import amp


def run(args):
    print('| Step: train_cam |')

    if not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    print('Loading dataset')
    dataset = ImageFolder(
        args.dataset_dir,
        transform=transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_threads,
        pin_memory=args.cuda,
        sampler=ImbalancedDatasetSampler(dataset))

    print('Building model')
    model = model_factory.get_model(args)
    if os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint), strict=False)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()
        criterion.cuda()

    # Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # Remove lr_decay events that have occured
    args.cur_iter = int(args.start_epoch * len(dataloader))
    while args.lr_decay_iters:
        if args.lr_decay_iters[0] <= args.cur_iter:
            args.lr_decay_iters.pop(0)
            print('Scaled learning rate by %s' % args.lr_decay)
            continue
        break

    # cuDNN looks for the optimal set of algorithms for network configuration
    # Benchmark mode is good whenever input sizes do not vary
    torch.backends.cudnn.benchmark = True

    # Initialize Nvidia/apex for mixed prec training
    print('Preparing model for mixed precision training')
    model, optimizer = amp.initialize(model, optimizer,
        opt_level=args.mixed_prec_level)
    print()

    print('Beginning training')
    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()

        print('Epoch: %s' % epoch)

        # Train for one epoch
        train(model, dataloader, criterion, optimizer, epoch, args)

        epoch_time = (time.time() - start) / 60 / 60
        print('Epoch time: %.2f hours\n' %  epoch_time)

        filename = 'checkpoint_%s-%i.pth.tar' % (args.architecture, epoch + 1)
        torch.save(model.state_dict(), os.path.join(args.model_save_dir, filename))


def train(model, dataloader, criterion, optimizer, epoch, args):
    """Train model on training set"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = MovingAverageMeter(alpha=0.999)
    top1 = MovingAverageMeter(alpha=0.997)

    end = time.time()
    for i, (input, target) in enumerate(dataloader):
        data_time.update(time.time() - end)
        args.cur_iter += 1

        if args.cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # Model inference
        output = model(input)

        # Compute output statistics
        loss = criterion(output, target)
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        losses.update(loss.data, input.size(0))

        # Compute gradient and step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # Update learning rate
        if args.lr_decay_iters:
            if args.lr_decay_iters[0] < args.cur_iter:
                args.lr_decay_iters.pop(0)
                args.lr = args.lr * args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                print('Learning rate adjusted to: %s' % args.lr)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Info log every print_freq
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

        if i % 1000 == 0:
            filename = 'checkpoint_%s-%i.pth.tar' % (args.architecture, epoch + 1)
            torch.save(model.state_dict(), os.path.join(args.model_save_dir, filename))
