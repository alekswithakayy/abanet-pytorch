import argparse
import os
import cv2
import sys
import shutil
import torch

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from models import model_factory
from util import load_checkpoint
from prm import PeakResponseMapping

from os.path import isfile, isdir, join, splitext
from collections import OrderedDict
from tqdm import tqdm


IMAGE_EXTENSIONS = ['.jpeg', '.jpg', '.png']
VIDEO_EXTENSIONS = ['.mp4']


def run(infer_args, model_args):

    ##############
    # Initialize #
    ##############

    print('** Initializing inference engine **')

    infer_args.cuda = torch.cuda.is_available()

    for key, value in vars(infer_args).items():
        print('{:20s}{:s}'.format(key, str(value)))
    print()

    infer_args.classes = sorted(
        [l.strip() for l in open(infer_args.class_list, 'r').readlines()])


    ###############
    # Build Model #
    ###############

    print('** Building model **')

    for key, value in vars(model_args).items():
        print('{:20s}{:s}'.format(key, str(value)))
    print()

    # Define model
    model = model_factory.get_model(model_args, infer_args.cuda)

    # Attempt to load model from checkpoint
    if model_args.checkpoint and os.path.isfile(model_args.checkpoint):
        print('Checkpoint found at: %s' % model_args.checkpoint)
        model, _, _ = load_checkpoint(model, model_args.checkpoint,
            infer_args.cuda)
        print('Checkpoint successfully loaded')
    else:
        print('No checkpoint found at: %s' % model_args.checkpoint)
        print('Halting inference')
        return
    print()


    ################
    # Analyze Data #
    ################

    print('** Beginning inference **')

    model.eval()

    for item in os.listdir(infer_args.inference_dir):
        print('Processing: %s' % item)

        item_name, ext = splitext(item)
        item_path = join(infer_args.inference_dir, item)

        results_file = open(join(infer_args.results_dir, item_name + '.csv'), 'w')

        if ext == '' and not 'DS_Store':
            results = process_directory(item_path, model, infer_args)

            frame_count = 0
            for result in results:
                results_file.write('%i,%s\n' % (frame_count, result))
                frame_count += 1
            print('\n')

        elif ext.lower() in VIDEO_EXTENSIONS:
            tmp_proc_dir = join(infer_args.inference_dir, 'tmp_proc_dir')
            if isdir(tmp_proc_dir): shutil.rmtree(tmp_proc_dir)
            os.mkdir(tmp_proc_dir)
            results = process_video(item_path, tmp_proc_dir, model, infer_args)
            shutil.rmtree(tmp_proc_dir)

            frame_count = 0
            for result in results:
                for _ in range(0, infer_args.every_nth_frame):
                    results_file.write(
                        '%i,%s\n' % (frame_count, result))
                    frame_count += 1
            print('\n')

        elif ext.lower() in IMAGE_EXTENSIONS:
            result = process_image(item_path, model, infer_args)
            results_file.write('%s' % result)
            print('\n')

        else:
            print('%s is not a recognized file type, skipping...' % item)
            continue

        results_file.close()


def process_image(image_path, model, infer_args):
    image_name, ext = splitext(image_path)

    if not ext.lower() in IMAGE_EXTENSIONS:
        print('%s is not a recognized image type, skipping...' % image_path)
        return

    image = load_image(image_path)
    input = transform_image(image, infer_args.image_size)

    if infer_args.prm:
        model = PeakResponseMapping(model)
        logits, crms, peaks, prms = model(input)
        _, idx = torch.max(logits, dim=1)
        idx = idx.item()
        class_ = infer_args.classes[idx]
        visualize_prm(image, crms[0, idx], prms, class_, infer_args.results_dir, image_path)
        return class_
    else:
        # Get species
        output = model(input).squeeze()
        _, idx = torch.max(output, dim=0)
        idx = idx.item()
        class_ = infer_args.classes[idx]
        return class_


def process_directory(dir_path, model, infer_args):
    print('Processing frames...')
    results = []
    files = os.listdir(dir_path)
    pbar = tqdm(total=len(files))
    for f in files:
        f_path = join(dir_path, f)
        result = process_image(f_path, model, infer_args)
        results.append(result)
        pbar.update()
    pbar.close()
    return results


def process_video(video_path, tmp_proc_dir, model, infer_args):
    print('Extracting frames from video...')

    _, ext = splitext(video_path)
    if not ext.lower() in VIDEO_EXTENSIONS:
        print('%s is not a recognized video type, skipping...' % path)
        return

    frame_count = 0
    video_cap = cv2.VideoCapture(video_path)
    while video_cap.isOpened():
        isvalid, frame = video_cap.read()
        if not (frame_count % infer_args.every_nth_frame == 0):
            frame_count += 1
            continue
        if isvalid:
            frame_path = join(tmp_proc_dir, 'frame_%i.png' % frame_count)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        else:
            break
    video_cap.release()

    return process_directory(tmp_proc_dir, model, infer_args)


def load_image(path):
    with open(path, 'rb') as f:
        image = Image.open(f)
        return image.convert('RGB')


def transform_image(image, size):
    if size:
        image = transforms.Resize((size, size), Image.ANTIALIAS)(image)
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image


# Archiving this method here for now
def visualize_prm(image, crm, prms, class_, results_dir, image_path):
        f, axarr = plt.subplots(3, 3, figsize=(7,7))

        # Display input image
        axarr[0,0].imshow(image)
        axarr[0,0].set_title('Image', size=6)
        axarr[0,0].axis('off')

        # Display class response maps
        axarr[0,1].imshow(crm.cpu(), interpolation='bicubic')
        axarr[0,1].set_title('Class Response ("%s")' % class_, size=6)
        axarr[0,1].axis('off')

        count = 0
        for i in range(1,3):
            for j in range(0,3):
                if count < len(prms):
                    axarr[i, j].imshow(prms[count].cpu(), cmap=plt.cm.jet)
                    axarr[i, j].set_title('Peak Response ("%s")' % class_, size=6)
                    count += 1
                axarr[i,j].axis('off')

        filename = os.path.basename(image_path)
        filename, _ = filename.rsplit('.', 1)
        plt.savefig(os.path.join(results_dir, filename) + '.png', dpi=300)
        plt.close()
