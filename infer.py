import argparse
import os
import cv2
import sys
import shutil
import torch
import scipy

import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from models import model_factory
from util import load_checkpoint
from prm import PeakResponseMapping

from os.path import isfile, isdir, join, splitext, basename
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
    infer_args.backgnd_idx = infer_args.classes.index('background')

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

    dir_items = os.listdir(infer_args.inference_dir)
    for i, item in enumerate(dir_items):
        print('Processing: %s (%i/%i)' % (item, i+1, len(dir_items)))

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
            #_process_video(item_path, model, infer_args)
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
            print('%s is not a recognized file type, skipping...\n' % item)
            continue

        results_file.close()


def process_image(image_path, model, infer_args):
    image_name, ext = splitext(image_path)

    if not ext.lower() in IMAGE_EXTENSIONS:
        print('%s is not a recognized image type, skipping...' % image_path)
        return

    image = load_image(image_path)
    image = transform_image(image, infer_args.image_size, infer_args.crop)
    input = transforms.ToTensor()(image).unsqueeze(0)
    if infer_args.cuda:
        input = input.cuda(async=True)

    if infer_args.visualize_results:
        logits, activation_maps = model(input)
        class_ = visualize_activation_map(image, logits, activation_maps, image_path, infer_args)
        return class_
    else:
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

    filename = basename(video_path)
    filename, ext = splitext(filename)
    if not ext.strip().lower() in VIDEO_EXTENSIONS:
        print('%s is not a recognized video type, skipping...' % video_path)
        return

    frame_count = 0
    video_cap = cv2.VideoCapture(video_path)
    while video_cap.isOpened():
        isvalid, frame = video_cap.read()
        if not (frame_count % infer_args.every_nth_frame == 0):
            frame_count += 1
            continue
        if isvalid:
            frame_path = join(tmp_proc_dir, filename + 'frame_%i.png' % frame_count)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        else:
            break
    video_cap.release()

    return process_directory(tmp_proc_dir, model, infer_args)


def _process_video(video_path, model, infer_args):
    print('Extracting frames from video...')

    filename = basename(video_path)
    filename, ext = splitext(filename)
    if not ext.strip().lower() in VIDEO_EXTENSIONS:
        print('%s is not a recognized video type, skipping...' % video_path)
        return

    video_cap = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0
    while video_cap.isOpened():
        isvalid, frame = video_cap.read()
        if not (frame_count % infer_args.every_nth_frame == 0):
            frame_count += 1
            continue
        if isvalid:
            frames.append(frame)
            frame_count += 1
        else:
            break
    video_cap.release()

    logits = []
    activation_maps = []
    batch = []
    for i, frame in enumerate(frames):
        image = Image.fromarray(frame)
        image = transform_image(image, infer_args.image_size, infer_args.crop)
        input = transforms.ToTensor()(image)
        if infer_args.cuda:
            input = input.cuda(async=True)
        batch.append(input)

        if (len(batch) == infer_args.batch_size) or (i == len(frames) - 1):
            batch = torch.stack(batch)
            l, am = model(batch)
            logits.append(l.detach().cpu().numpy())
            activation_maps.append(am.detach().cpu().numpy())
            batch = []

    if logits[-1].ndim == 1:
        logits[-1] = np.expand_dims(logits[-1], 0)
    logits = np.concatenate(logits)
    activation_maps = np.concatenate(activation_maps)


    class_per_frame = [-1]*len(frames)
    top3_per_frame = np.flip(np.argsort(logits, axis=1)[:,-3:], axis=1)
    scores = [0]*len(infer_args.classes)
    for i, top3 in enumerate(top3_per_frame):
        if top3[0] == infer_args.backgnd_idx:
            class_per_frame[i] = infer_args.backgnd_idx
            continue
        score = 3
        for j in top3.tolist():
            if not j == infer_args.backgnd_idx:
                scores[j] += score
            score -= 1
    max_score = max(scores)
    if max_score == 0:
        class_idx = infer_args.backgnd_idx
    else:
        class_idx = scores.index(max_score)
    class_per_frame = [class_idx if i == -1 else i for i in class_per_frame]
    print(scores)
    print(infer_args.classes[class_idx])

    class_activation_maps = activation_maps[:,class_idx,:,:].squeeze()
    f, x, y = class_activation_maps.shape

    for i in range(f):
        class_activation_maps[i,:,:] = blur(class_activation_maps[i,:,:])
    # mean = class_activation_maps.reshape(f*x*y).mean(axis=0)
    # std = class_activation_maps.reshape(f*x*y).std(axis=0)
    # class_activation_maps = class_activation_maps > (mean + 1.0 * std)
    # class_activation_maps = class_activation_maps.astype(int)

    # Animal presence
    presence = class_activation_maps.reshape(f, x*y).mean(axis=1).tolist()

    # Volatility
    diff = np.abs(class_activation_maps[1:,:,:] - class_activation_maps[:-1,:,:])
    volatility = diff.reshape(f-1, x*y).mean(axis=1).tolist()
    volatility.insert(0, 0.0)

    if infer_args.visualize_results:
        create_video_visualization(video_path, class_per_frame, presence, volatility, infer_args)


def create_video_visualization(video_path, class_per_frame, presence, volatility, infer_args):
    video_cap = cv2.VideoCapture(video_path)
    length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    name, ext = splitext(basename(video_path))
    new_video_path = join(infer_args.results_dir, name + '_result' + ext.lower())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    results_video = cv2.VideoWriter(new_video_path, fourcc, fps, (width, height))

    font_scale = 0.8
    font_color = (0,0,255)
    line_type = 2

    frame_count = 0
    while video_cap.isOpened():
        isvalid, frame = video_cap.read()
        if isvalid and class_per_frame:
            if frame_count % infer_args.every_nth_frame == 0:
                i = class_per_frame.pop(0)
                p = presence.pop(0)
                v = volatility.pop(0)
                frame_count += 1
            text = []
            text.append('Class=%s' % infer_args.classes[i])
            text.append('Presence=%.3f' % p)
            text.append('Volatility=%.3f' % v)
            text_loc = (20,60)
            for t in text:
                cv2.putText(frame, t, text_loc, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, font_color, line_type)
                text_loc = (text_loc[0], text_loc[1] + 50)
            frame_count += 1
        else:
            break
        results_video.write(frame)

    video_cap.release()
    results_video.release()
    cv2.destroyAllWindows()


def visualize_activation_map(image, logits, activation_maps, image_path, infer_args):
    _, idx = torch.max(logits.squeeze(), dim=0)
    class_idx = idx.item()
    class_ = infer_args.classes[class_idx]
    backgnd_idx = infer_args.classes.index('background')

    activation_maps = activation_maps.detach()
    b, c, h, w = activation_maps.size()
    O_c = F.softmax(activation_maps, dim=1)
    M_c = O_c * torch.sigmoid(activation_maps)
    alpha_c = F.softmax(M_c.view(b, c, h*w), dim=2)
    activation_maps = alpha_c.view(b, c, h, w) * activation_maps
    activation_maps = activation_maps.squeeze()

    class_map = activation_maps[class_idx,:,:]
    backgnd_map = activation_maps[backgnd_idx,:,:]

    _, axarr = plt.subplots(1, 3, figsize=(12,4))

    # Display input image
    axarr[0].imshow(image)
    axarr[0].set_title('Image', size=6)
    axarr[0].axis('off')

    # Display class response map
    axarr[1].imshow(class_map.cpu(), interpolation='bicubic')
    axarr[1].set_title('Class Response ("%s")' % class_, size=6)
    axarr[1].axis('off')

    # Display background response map
    axarr[2].imshow(backgnd_map.cpu(), interpolation='bicubic')
    axarr[2].set_title('Class Response ("background")', size=6)
    axarr[2].axis('off')

    filename = basename(image_path)
    filename, _ = filename.rsplit('.', 1)
    plt.savefig(join(infer_args.results_dir, filename) + '.png', dpi=300)
    plt.close()

    return class_


def load_image(path):
    with open(path, 'rb') as f:
        image = Image.open(f)
        return image.convert('RGB')


def transform_image(image, size, crop):
    if crop:
        image = image.crop(crop)
    if size:
        image = transforms.Resize(size, Image.ANTIALIAS)(image)
    return image

def blur(a):
    kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
    kernel = kernel / np.sum(kernel)
    arraylist = []
    for y in range(3):
        temparray = np.copy(a)
        temparray = np.roll(temparray, y - 1, axis=0)
        for x in range(3):
            temparray_X = np.copy(temparray)
            temparray_X = np.roll(temparray_X, x - 1, axis=1)*kernel[y,x]
            arraylist.append(temparray_X)

    arraylist = np.array(arraylist)
    arraylist_sum = np.sum(arraylist, axis=0)
    return arraylist_sum
