import os
import time
import numpy as np
import imageio
import torch

from torch import multiprocessing
from torch.utils.data import DataLoader, Subset

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

from datasets.custom_image_folders import MakeIRLabelImageFolder
from util import AverageMeter, MovingAverageMeter

import matplotlib.pyplot as plt


def run(args):
    print('| Step: make_ir_label |')

    if not os.path.isdir(args.ir_label_save_dir):
        os.makedirs(args.ir_label_save_dir)

    print('Loading dataset')
    dataset = MakeIRLabelImageFolder(args)
    dataset = [Subset(dataset, np.arange(i, len(dataset), args.num_threads))
               for i in range(args.num_threads)]

    print('Processing labels')
    multiprocessing.spawn(_work, nprocs=args.num_threads, args=(dataset, args), join=True)


def _work(process_id, dataset, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for i, (image, cam, target, name) in enumerate(dataset[process_id]):
        data_time.update(time.time() - end)

        keys = np.pad(np.array([target]) + 1, (1, 0), mode='constant')

        # Find confident fg
        fg_conf_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.60)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        fg_pred = crf_inference(np.uint8(image*255), fg_conf_cam, t=12, n_labels=2)
        fg_pred *= target + 1

        # Find confident bg
        bg_conf_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.40)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        bg_pred = crf_inference(np.uint8(image*255), bg_conf_cam, t=12, n_labels=2)

        # Combine labels into confident fg, confident bg and not confident bg
        conf = fg_pred.copy()
        conf[fg_pred == 0] = 255
        conf[bg_pred + fg_pred == 0] = 0

        imageio.imwrite(os.path.join(args.ir_label_save_dir, name + '.png'), conf.astype(np.uint8))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   i, len(dataset[process_id]), batch_time=batch_time,
                   data_time=data_time))

            # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
            # ax1.imshow(image)
            # ax2.imshow(conf)
            # ax3.imshow(cam.squeeze())
            # plt.savefig(os.path.join('/home/adjuric/data/abanet/' + 'plots_' + args.architecture, name))
            # plt.close()


def crf_inference(image, labels, t=10, n_labels=21, gt_prob=0.7):
    h, w = image.shape[:2]
    crf = dcrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    crf.setUnaryEnergy(unary)
    crf.addPairwiseGaussian(sxy=3, compat=3)
    crf.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(image)), compat=10)
    q = crf.inference(t)
    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
