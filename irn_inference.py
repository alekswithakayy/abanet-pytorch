import os
import time
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models import model_factory
from models.irn import EdgeDisplacement
from util import AverageMeter

import matplotlib.pyplot as plt
from apex import amp

def run(args):
    print('| Step: irn_inference |')

    if not os.path.isdir(args.cam_save_dir):
        os.makedirs(args.cam_save_dir)

    print('Loading dataset')
    dataset = ImageFolder(args.dataset_dir,
        transform=transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=1,
        num_workers=args.num_threads, pin_memory=args.cuda)

    print('Building model')
    backbone = model_factory.get_model(args)
    model = EdgeDisplacement(backbone, args)
    if os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint), strict=False)
    else:
        raise ValueError('No model checkpoint found at %s' % args.checkpoint)

    if args.cuda:
        model.cuda()

    # Initialize Nvidia/apex for mixed prec training
    print('Preparing model for mixed precision training')
    model = amp.initialize(model, opt_level='O2')

    print('Beginning IRN inference')
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(dataloader):
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda(non_blocking=True)

        # Model inference
        edge, df = model(input)

        image = np.uint8(np.transpose(input[0].detach().cpu().numpy(), (1,2,0))*255)
        edge = np.transpose(edge[0].detach().cpu().numpy(), (1,2,0)).squeeze()
        df = np.transpose(df[0].detach().cpu().numpy(), (1,2,0))
        df = np.pad(df, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
        df = df / np.max(np.abs(df))

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        ax1.imshow(image)
        ax2.imshow(edge)
        ax3.imshow(df)
        plt.savefig(os.path.join('/home/adjuric/data/abanet/' + 'plots_' + args.architecture, str(i)))
        plt.close()

        batch_time.update(time.time() - end)
        end = time.time()
