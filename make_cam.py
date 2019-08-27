import numpy as np
import os
import time
import torch

from torch.utils.data import DataLoader
from datasets.custom_image_folders import MakeCAMImageFolder
from models import model_factory
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from util import AverageMeter

from apex import amp

def run(args):
    print('| Step: make_cam |')

    if not os.path.isdir(args.cam_save_dir):
        os.makedirs(args.cam_save_dir)

    print('Loading dataset')
    dataset = MakeCAMImageFolder(args)

    print('Building model')

    # Define model
    model = model_factory.get_model(args)
    if os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint), strict=True)
    else:
        raise ValueError('No model checkpoint found at %s' % args.checkpoint)

    if args.cuda:
        model.cuda()

    # Initialize Nvidia/apex for mixed prec training
    print('Preparing model for mixed precision training')
    model = amp.initialize(model, opt_level='O2')

    print('Beginning CAM inference')
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for i, (input, target, name) in enumerate(dataset):
        data_time.update(time.time() - end)

        if args.cuda:
            input = [x.cuda(non_blocking=True) for x in input]

        cams = [model(image.unsqueeze(0))[1] for image in input]
        cams = [F.interpolate(cam, args.image_size, mode='bilinear') for cam in cams]

        merged_cam = torch.sum(torch.stack(cams, 0), 0)
        merged_cam = merged_cam[:,target]
        merged_cam /=  F.adaptive_max_pool2d(merged_cam, (1, 1)).max() + 1e-5
        merged_cam = merged_cam.detach().cpu().numpy()

        np.save(os.path.join(args.cam_save_dir, name + '.npy'), merged_cam)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        fg_conf_cam = np.pad(merged_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.6)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        image = np.uint8(np.transpose(input[0].detach().cpu().numpy(), (1,2,0))*255)
        pred = crf_inference(image, fg_conf_cam, t=10, n_labels=2)

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        ax1.imshow(image)
        ax2.imshow(pred)
        ax3.imshow(merged_cam.squeeze())
        plt.savefig(os.path.join('/home/adjuric/data/abanet/' + 'plots_' + args.architecture, name))
        plt.close()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   i, len(dataset), batch_time=batch_time,
                   data_time=data_time))


import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax
def crf_inference(img, labels, t=10, n_labels=21, gt_prob=0.7):
    h, w = img.shape[:2]
    crf = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    crf.setUnaryEnergy(unary)
    crf.addPairwiseGaussian(sxy=3, compat=3)
    crf.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)
    q = crf.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
