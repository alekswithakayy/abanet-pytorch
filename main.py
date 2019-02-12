import argparse
import numpy
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms

from torch.optim import SGD
from torch.utils.data import DataLoader

from util import sampler
from datasets import dataset_factory
from models import model_factory

from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import imresize


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

parser.add_argument('--start_epoch',
                    default=-1,
                    type=int,
                    help='Epoch to start training at (effects learning rate)')

parser.add_argument('--train',
                    default=True,
                    type=str2bool,
                    help='Train the model')

parser.add_argument('--evaluate',
                    default=True,
                    type=str2bool,
                    help='Evaluate model on validation set')

parser.add_argument('--inference',
                    default=False,
                    type=str2bool,
                    help='Use model to perform inference on dataset')

parser.add_argument('--inference_dir',
                    default='/data',
                    help='Directory containing images to be inferenced.')

parser.add_argument('--trainable_params',
                    required=False,
                    nargs='+',
                    help='List of trainable params. ex: Including layer1 will'
                         'train all model params that contain layer1 in their'
                         'name. Use model.named_parameters() to see all names.')

parser.add_argument('--randomize_params',
                    default=[],
                    required=False,
                    nargs='+',
                    help='List of params to randomize in model. Parameters '
                         'that are not randomized will be restored from the '
                         'checkpoint.')

parser.add_argument('--pretrained',
                    default=True,
                    type=str2bool,
                    help='Use pre-trained model.')


def main():

    ##############
    # Initialize #
    ##############

    global args, best_prec1, cuda, classes

    args = parser.parse_args()

    if not os.path.isdir(args.model_dir): os.makedirs(args.model_dir)

    # Check if cuda is available
    cuda = torch.cuda.is_available()
    print("Using cuda: %s" % cuda)

    #torch.set_printoptions(threshold=10000)

    ###################
    # Create Datasets #
    ###################

    # Create train loader
    train_dir = os.path.join(args.dataset_dir, 'train')
    train_dataset = dataset_factory.get_dataset(args.dataset_name, 'train',
                                                train_dir, args.image_size)
    classes = train_dataset.classes
    train_sampler = sampler.ImbalancedDatasetSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
        num_workers=args.num_threads, pin_memory=cuda, sampler=train_sampler)

    # Create validation loader
    val_dir = os.path.join(args.dataset_dir, 'val')
    val_dataset = dataset_factory.get_dataset(args.dataset_name, 'val', val_dir,
                                              args.image_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_threads, pin_memory=cuda)


    ###############
    # Build Model #
    ###############

    model = model_factory.get_model(args.model_arch, len(classes),
                                    args.pretrained)

    if args.trainable_params:
        print('Searching for parameter names containing: %s' % args.trainable_params)

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze and collect all trainable parameters
        trainable_params = []
        trainable_param_count = 0
        for name, param in model.named_parameters():
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
            from collections import OrderedDict
            if cuda:
                checkpoint = torch.load(args.checkpoint)
                # try:
                #     optimizer.load_state_dict(checkpoint['optimizer'])
                #     for state in optimizer.state.values():
                #         for k, v in state.items():
                #             if isinstance(v, torch.Tensor): state[k] = v.cuda()
                # except ValueError:
                #     print('Could not load optimizer state_dict')
                new_state_dict = OrderedDict()
                for name, param in checkpoint['state_dict'].items():
                    add_param = True
                    for word in args.randomize_params:
                        if word in name:
                            add_param = False
                            break
                    if add_param: new_state_dict[name] = param
                model.load_state_dict(new_state_dict, strict=False)
            else:
                checkpoint = torch.load(args.checkpoint,
                                        map_location=lambda storage, loc: storage)
                # try:
                #     optimizer.load_state_dict(checkpoint['optimizer'])
                # except ValueError:
                #     print('Could not load optimizer state_dict')
                new_state_dict = OrderedDict()
                for name, param in checkpoint['state_dict'].items():
                    # remove 'module.' of dataparallel
                    if name.startswith('module.'): name = name[7:]
                    if name.startswith('0.'): name = name[2:]
                    add_param = True
                    for word in args.randomize_params:
                        if word in name:
                            add_param = False
                            break
                    if add_param: new_state_dict[name] = param
                model.load_state_dict(new_state_dict, strict=False)
            start_epoch = checkpoint['epoch']
            print('Loaded checkpoint at epoch: %i' % start_epoch)
        else:
            print('No checkpoint found at: %s' % args.checkpoint)
    else:
        start_epoch = 0

    if args.start_epoch < 0:
        args.start_epoch = start_epoch
    print('Beginning training from epoch: %s' % args.start_epoch)

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
            lr = adjust_learning_rate(optimizer, epoch)
            print('Learning rate: %s' % lr)

            # Train for one epoch
            train(model, train_loader, criterion, optimizer, epoch)

            state = {
                'epoch': epoch + 1,
                'model_arch': args.model_arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }

            filename = 'checkpoint_%s-%i.pth.tar' % (args.model_arch, epoch + 1)
            torch.save(state, os.path.join(args.model_dir, filename))


    ##############
    # Evaluation #
    ##############

    if args.evaluate:
        print("Evaluating model...")
        validate(model, val_loader, criterion)
        return


    #############
    # Inference #
    #############

    if args.inference:
        print("Model inferencing...")
        inference(model)


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


def inference(model):
    #model.inference()
    model.train(False)
    resize_crop = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size)
    ])

    for filename in os.listdir(args.inference_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = Image.open(os.path.join(args.inference_dir, filename)).convert('RGB')
            image = resize_crop(image)
            input = transforms.ToTensor()(image).unsqueeze(0)

            if cuda:
                input = input.cuda().requires_grad_()
            else:
                input.requires_grad_()

            output = model(input)

            if output:
                # Confidence, Class Response Maps,
                # Class Peak Response, Peak Response Maps
                # conf, crms, cprs, prms = output
                conf, crms = output
                crms = crms.detach()
                _, idx = torch.max(conf, dim=1)
                idx = idx.item()

                print('Class index: %i, Class: %s' % (idx, classes[idx]))
                # print(cprs.size())

                f, axarr = plt.subplots(3, 3, figsize=(7,7))

                # Display input image
                axarr[0,0].imshow(image)
                axarr[0,0].set_title('Image', size=6)
                axarr[0,0].axis('off')

                # Display class response maps
                axarr[0,1].imshow(crms[0, idx].cpu(), interpolation='bicubic')
                axarr[0,1].set_title('Class Response ("%s")' % classes[idx], size=6)
                axarr[0,1].axis('off')

                axarr[0,2].imshow(crms[0, 4].cpu(), interpolation='bicubic')
                axarr[0,2].set_title('Class Response ("%s")' % classes[4], size=6)
                axarr[0,2].axis('off')

                # Display peak response maps
                # count = 0
                # for i in range(1,3):
                #     for j in range(0,3):
                #         if count < len(cprs):
                #             axarr[i, j].imshow(prms[count].cpu(), cmap=plt.cm.jet)
                #             axarr[i, j].set_title('Peak Response ("%s")' % (classes[idx]), size=6)
                #             count += 1
                #         axarr[i,j].axis('off')
                # filename, _ = filename.split('.')
                plt.savefig(os.path.join('/Users/aleksandardjuric/Desktop/', filename) + '.png', dpi=300)
                # plt.show()
                plt.close()
            else:
                print('No class peak response detected for %s' % os.path.basename(filename))


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
