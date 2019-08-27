import os
import time
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from util import indexing

from models import model_factory
from datasets.custom_image_folders import TrainIRNImageFolder
from util.sampler import ImbalancedDatasetSampler
from util import AverageMeter

def run(args):
    print('| Step: train_irn |')

    if not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    path_index = indexing.PathIndex(radius=10,
        size=(args.image_size[0] // 4, args.image_size[1] // 4))

    print('Loading dataset')
    dataset = TrainIRNImageFolder(path_index.src_indices, path_index.dst_indices, args)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_threads,
                            pin_memory=args.cuda,
                            sampler=ImbalancedDatasetSampler(dataset))

    print('Building model')
    from models.irn import AffinityDisplacement

    # backbone = model_factory.get_model(args)
    # if os.path.isfile(args.checkpoint):
    #     backbone.load_state_dict(torch.load(args.checkpoint), strict=True)
    #
    # model = AffinityDisplacement(backbone,
    #                              path_index.paths_by_len_indices,
    #                              torch.from_numpy(path_index.src_indices),
    #                              torch.from_numpy(path_index.dst_indices),
    #                              args)

    backbone = model_factory.get_model(args)
    model = AffinityDisplacement(backbone,
                                 path_index.paths_by_len_indices,
                                 torch.from_numpy(path_index.src_indices),
                                 torch.from_numpy(path_index.dst_indices),
                                 args)
    if os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint), strict=False)

    criterion = None

    if args.cuda:
        model.cuda()

    # Define optimizer
    edge_params, df_params = model.trainable_parameters()
    optimizer = torch.optim.SGD([{'params': edge_params, 'lr': 1*args.lr},
                                 {'params': df_params, 'lr': 10*args.lr}],
                                lr=args.lr, momentum=args.momentum,
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

    print('Beginning training')
    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()

        print('Epoch: %s' % epoch)

        # Train for one epoch
        train(model, dataloader, criterion, path_index, optimizer, epoch, args)

        epoch_time = (time.time() - start) / 60 / 60
        print('Epoch time: %.2f hours\n' %  epoch_time)

        filename = 'checkpoint_%s-%i.pth.tar' % (args.architecture, epoch + 1)
        torch.save(model.state_dict(), os.path.join(args.model_save_dir, filename))


def train(model, dataloader, criterion, path_index, optimizer, epoch, args):
    """Train model on training set"""

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (image, bg_pos, fg_pos, neg) in enumerate(dataloader):
        data_time.update(time.time() - end)
        args.cur_iter += 1

        if args.cuda:
            image = image.cuda(non_blocking=True)
            bg_pos = bg_pos.cuda(non_blocking=True)
            fg_pos = fg_pos.cuda(non_blocking=True)
            neg = neg.cuda(non_blocking=True)

        # Model inference
        aff, df = model(image)

        df = path_index.to_displacement(df)

        bg_pos_aff_loss = torch.sum(-bg_pos * torch.log(aff + 1e-5)) / (torch.sum(bg_pos) + 1e-5)
        fg_pos_aff_loss = torch.sum(-fg_pos * torch.log(aff + 1e-5)) / (torch.sum(fg_pos) + 1e-5)
        pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
        neg_aff_loss = torch.sum(-neg * torch.log(1. + 1e-5 - aff)) / (torch.sum(neg) + 1e-5)

        df_fg_loss = torch.sum(path_index.to_displacement_loss(df) * torch.unsqueeze(fg_pos, 1)) / (2*torch.sum(fg_pos) + 1e-5)
        df_bg_loss = torch.sum(torch.abs(df) * torch.unsqueeze(bg_pos, 1)) / (2*torch.sum(bg_pos) + 1e-5)

        total_loss = (pos_aff_loss + neg_aff_loss)/2 + (df_fg_loss + df_bg_loss)/2

        losses.update(total_loss.data, image.size(0))

        # Compute gradient and step
        optimizer.zero_grad()
        total_loss.backward()
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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(dataloader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

        if i % 1000 == 0:
            filename = 'checkpoint_%s-%i.pth.tar' % (args.architecture, epoch + 1)
            torch.save(model.state_dict(), os.path.join(args.model_save_dir, filename))
