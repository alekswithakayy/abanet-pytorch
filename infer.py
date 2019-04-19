import argparse
import os
import cv2
import sys
import shutil
import torch
import skimage
import time

import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from models import model_factory
from util import load_checkpoint
from prm import PeakResponseMapping

from os.path import isfile, isdir, join, splitext, basename
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from apex import amp

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

    # Initialize Nvidia/apex for mixed prec training
    print('Preparing model for mixed precision training')
    model = amp.initialize(model, opt_level='O2')
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

        if ext.lower() in IMAGE_EXTENSIONS:
            process_image(item_path, model, infer_args)
        elif ext.lower() in VIDEO_EXTENSIONS:
            process_video(item_path, model, infer_args)
        else:
            print('%s is not a recognized file type, skipping...\n' % item)
            continue

        results_file.close()


def process_image(image_path, model, infer_args):
    image_name = basename(image_path)
    image_name, ext = splitext(image_name)
    if not ext.lower() in IMAGE_EXTENSIONS:
        print('%s is not a recognized image type, skipping...' % image_name)
        return

    image = load_image(image_path)
    image = crop(image, infer_args.crop)
    image = resize(image, infer_args.image_size)
    input = transforms.ToTensor()(image).unsqueeze(0).contiguous().float()
    if infer_args.cuda:
        input = input.cuda(async=True)

    logits, act_maps = model(input)
    _, top1_idx = torch.max(logits, dim=0)
    top1_idx = top1_idx.item()
    top1_class = infer_args.classes[top1_idx]

    top1_act_map = act_maps[0,top1_idx,:,:].detach().cpu().numpy()
    top1_act_map = resize(top1_act_map, infer_args.image_size)
    thresh = skimage.filters.threshold_otsu(top1_act_map)
    top1_act_map = top1_act_map > thresh

    if infer_args.visualize_results:
        visualize_image_results(image_path, top1_class, top1_act_map, infer_args)


def visualize_image_results(image_path, top1_class, top1_act_map, infer_args):
    image = load_image(image_path)
    image = crop(image, infer_args.crop)

    h, w, c = image.shape
    map = resize(top1_act_map, (h, w), order=0)
    overlay = np.zeros((h, w, c), dtype=np.bool_)
    for i in range(3):
        overlay[:,:,i] = map
    overlay = np.uint8(np.copy(image) * np.invert(overlay) + overlay * 255)
    image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    plt.figure()
    plt.imshow(image)
    plt.title('%s' % top1_class, size=6)
    plt.axis('off')

    filename = basename(image_path)
    filename, _ = filename.rsplit('.', 1)
    plt.savefig(join(infer_args.results_dir, filename) + '.png', dpi=300)
    plt.close()


def process_video(video_path, model, infer_args):
    video_name = basename(video_path)
    video_name, ext = splitext(video_name)
    if not ext.strip().lower() in VIDEO_EXTENSIONS:
        print('%s is not a recognized video type, skipping...' % video_name)
        return

    frames = extract_frames(video_path, infer_args)
    frames = preprocess_frames(frames, infer_args)

    logits = []
    act_maps = []
    with torch.no_grad():
        to_tensor = transforms.ToTensor()
        for i in range(0, len(frames), infer_args.batch_size):
            batch = frames[i:i+infer_args.batch_size]
            batch = [to_tensor(frame).contiguous().float() for frame in batch]
            batch = torch.stack(batch)
            if infer_args.cuda:
                batch = batch.cuda(async=True)
            logit, act_map = model(batch)
            logits.append(logit.detach().cpu().numpy())
            act_maps.append(act_map.detach().cpu().numpy())

    if logits[-1].ndim == 1:
        logits[-1] = np.expand_dims(logits[-1], 0)
    logits = np.concatenate(logits)
    frame_labels, top1_idx = label_frames(logits, infer_args)

    act_maps = np.concatenate(act_maps)
    top1_act_maps = act_maps[:,top1_idx,:,:].squeeze()
    top1_act_maps = interpolate_activation_maps(top1_act_maps, infer_args)

    thresh = skimage.filters.threshold_otsu(top1_act_maps)
    top1_act_maps = np.uint8(top1_act_maps > thresh)

    f,x,y = top1_act_maps.shape
    presence = top1_act_maps.reshape(f, x*y).mean(axis=1).tolist()
    delta = np.abs(top1_act_maps[1:,:,:] - top1_act_maps[:-1,:,:])
    volatility = delta.reshape(f-1, x*y).mean(axis=1).tolist()
    volatility.insert(0, 0.0)

    if infer_args.visualize_results:
        visualize_video_results(video_path, top1_act_maps, frame_labels,
            presence, volatility, infer_args)


def extract_frames(video_path, infer_args):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while video_capture.isOpened():
        isvalid, frame = video_capture.read()
        if not (frame_count % infer_args.every_nth_frame == 0):
            frame_count += 1
            continue
        if isvalid:
            frame = frame[:,:,::-1]
            frames.append(frame)
            frame_count += 1
        else:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    return frames


def _preprocess_frame(i, frame, processed, infer_args):
    frame = crop(frame, infer_args.crop)
    if infer_args.six_crop:
        sections = [resize(c, infer_args.image_size) for c in six_crop(frame)]
        processed[i] = sections
    else:
        frame = resize(frame, infer_args.image_size)
        processed[i] = [frame]


def preprocess_frames(frames, infer_args):
    processed = [[]]*len(frames)
    with ThreadPoolExecutor(max_workers=infer_args.num_threads) as executor:
        for i, frame in enumerate(frames):
            executor.submit(_preprocess_frame, i, frame, processed, infer_args)
    return [i for f in processed for i in f]


def label_frames(logits, infer_args):
    class_scores = [0]*len(infer_args.classes)
    top1_per_image = [-1]*len(logits)
    top3_per_image = np.flip(np.argsort(logits, axis=1)[:,-3:], axis=1)
    for i, top3 in enumerate(top3_per_image):
        # If background is top1, immediately label image as background
        if top3[0] == infer_args.backgnd_idx:
            top1_per_image[i] = infer_args.backgnd_idx
            continue
        # Top scoring class = 3 points, second = 2 points, third = 1 point
        for j, idx in enumerate(top3.tolist()):
            if idx != infer_args.backgnd_idx:
                class_scores[idx] += 3 - j
    for i, score in enumerate(class_scores):
        if score: print('\t%s: %i' % (infer_args.classes[i], score))

    # Find index of overall top scoring class
    max_score = max(class_scores)
    if max_score == 0:
        top1_idx = infer_args.backgnd_idx
    else:
        top1_idx = class_scores.index(max_score)

    # Label each frame as top1_idx or backgnd_idx
    frame_labels = []
    if infer_args.six_crop:
        for i in range(0, len(top1_per_image), 6):
            if sum(top1_per_image[i:i+6]) / 6 == infer_args.backgnd_idx:
                frame_labels.append(infer_args.backgnd_idx)
            else:
                frame_labels.append(top1_idx)
    else:
        frame_labels = [top1_idx if i == -1 else i for i in top1_per_image]

    return frame_labels, top1_idx


def interpolate_activation_maps(maps, infer_args, order=1):
    size = infer_args.image_size
    interpolated_maps = []
    if infer_args.six_crop:
        section_size = [int(size[0]*0.6), int(size[1]*0.6)]
        for i in range(0, len(maps), 6):
            to_merge = []
            for map in maps[i:i+5]:
                to_merge.append(resize(map, section_size, order=order))
            to_merge.append(resize(maps[i+5], size, order=order))
            interpolated_maps.append(merge_six_crop(to_merge))
    else:
        for map in maps:
            map = resize(map, size, order=order)
            interpolated_maps.append(map)

    return np.array(interpolated_maps)


def visualize_video_results(video_path, top1_act_maps, class_per_frame,
    presence, volatility, infer_args):
    # Store video parameters
    video_capture = cv2.VideoCapture(video_path)
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    if infer_args.crop:
        x1, y1, x2, y2 = infer_args.crop
        width = x2 - x1
        height = y2 - y1
    else:
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize output video
    name, ext = splitext(basename(video_path))
    new_video_path = join(infer_args.results_dir, name + '_result' + ext.lower())
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(new_video_path, fourcc, fps, (width, height))

    # Edit frames
    frame_count = 0
    while video_capture.isOpened():
        isvalid, frame = video_capture.read()
        if isvalid and class_per_frame:
            if frame_count % infer_args.every_nth_frame == 0:
                idx = frame_count // infer_args.every_nth_frame
                # Create class attention overlay
                map = resize(top1_act_maps[idx], (height, width), order=0)
                overlay = np.zeros((height, width, 3), dtype=np.bool_)
                for i in range(3):
                    overlay[:,:,i] = map
                # Create presence and volatility plots
                cur_class = infer_args.classes[class_per_frame[idx]]
                plot = plot_as_array(idx, cur_class, presence, volatility)
            # Edit frame with overlay and plot
            frame = crop(frame, infer_args.crop)
            frame_overlay = np.uint8(np.copy(frame)*np.invert(overlay) + overlay*255)
            frame = cv2.addWeighted(frame, 0.6, frame_overlay, 0.4, 0)
            frame[-200:,-800:,:] = plot
            frame_count += 1
        else:
            break
        video_writer.write(frame)

    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()


def plot_as_array(i, cur_class, presence, volatility):
    fig = plt.figure(figsize=(8,2))

    presence_plot = fig.add_subplot(121)
    presence_plot.plot(range(i), presence[:i])
    presence_plot.set_title("Animal Presence (%s)" % cur_class)
    presence_plot.set_xlim(left=0, right=len(presence))
    presence_plot.set_ylim(bottom=0, top=max(presence)*1.25)

    volatility_plot = fig.add_subplot(122)
    volatility_plot.plot(range(i), volatility[:i])
    volatility_plot.set_title("Volatility")
    volatility_plot.set_xlim(left=0, right=len(volatility))
    volatility_plot.set_ylim(bottom=0, top=max(volatility)*1.25)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)

    # Pixmap in ARGB mode, roll the alpha channel to RGBA mode
    buf = np.roll(buf, 3, axis=2)
    buf = np.uint8(skimage.color.rgba2rgb(buf) * 255)
    plt.close()

    return buf


def load_image(path):
    image = skimage.io.imread(path)
    if image.shape[2] == 4:
        image = skimage.color.rgba2rgb(image)
    return image


def resize(image, size, order=1, anti_aliasing=True):
    if size:
        image = skimage.transform.resize(image, size, order=order,
            anti_aliasing=anti_aliasing)
    return image


def crop(image, crop):
    if crop:
        x1,y1,x2,y2 = crop
        image = image[y1:y2,x1:x2,:]
    return image


def six_crop(image):
    h, w, _ = image.shape
    crop_h = int(h * 0.6)
    crop_w = int(w * 0.6)
    tl = image[:crop_h,:crop_w,:]
    tr = image[:crop_h,-crop_w:,:]
    bl = image[-crop_h:,:crop_w,:]
    br = image[-crop_h:,-crop_w:,:]
    center = image[h//5:h//5+crop_h, w//5:w//5+crop_w,:]
    return [tl, tr, bl, br, center, image]


def merge_six_crop(images):
    tl, tr, bl, br, center, image = images
    h, w = image.shape
    crop_h = int(h * 0.6)
    crop_w = int(w * 0.6)

    image[:crop_h,:crop_w] += tl
    image[:crop_h,-crop_w:] += tr
    image[-crop_h:,:crop_w] += bl
    image[-crop_h:,-crop_w:] += br
    image[h//5:h//5+crop_h, w//5:w//5+crop_w] += center

    den = np.ones((h,w))
    den[:crop_h,:crop_w] += 1
    den[:crop_h,-crop_w:] += 1
    den[-crop_h:,:crop_w] += 1
    den[-crop_h:,-crop_w:] += 1
    den[h//5:h//5+crop_h, w//5:w//5+crop_w] += 1

    return image / den
