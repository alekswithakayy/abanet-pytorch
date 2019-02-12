import argparse
import os
import cv2
import sys
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms

from PIL import Image
from models import model_factory
from os.path import isfile, isdir, join, splitext
from collections import OrderedDict
from tqdm import tqdm

####################
# Input Parameters #
####################

parser = argparse.ArgumentParser()

parser.add_argument('--model_arch',
                    default='resnet_fcn',
                    type=str,
                    help='Name of model architecture')

parser.add_argument('--image_size',
                    default=448,
                    type=int,
                    help='Size of image.')

parser.add_argument('--checkpoint_species',
                    default='',
                    type=str,
                    help='Path to latest species model checkpoint.')

parser.add_argument('--class_list_species',
                    default='',
                    type=str,
                    help='Path to list of species classes.')

parser.add_argument('--checkpoint_counting',
                    default='',
                    type=str,
                    help='Path to latest counting model checkpoint.')

parser.add_argument('--class_list_counting',
                    default='',
                    type=str,
                    help='Path to list of counting classes.')

parser.add_argument('--results_dir',
                    default='/data',
                    help='Directory where results will be saved.')

parser.add_argument('--inference_dir',
                    default='/data',
                    help='Directory containing images/videos to be inferenced.')

parser.add_argument('--every_nth_frame',
                    default=30,
                    type=int,
                    help='Process every nth frame in a video.')


IMAGE_EXTENSIONS = ['.jpeg', '.jpg', '.png']
VIDEO_EXTENSIONS = ['.mp4']


def main():

    ##############
    # Initialize #
    ##############

    global args, species_model, counting_model

    args = parser.parse_args()

    # Check if cuda is available
    cuda = torch.cuda.is_available()
    print("Using cuda: %s" % cuda)

    args.class_list_species = sorted(
        [l.strip() for l in open(args.class_list_species, 'r').readlines()])

    args.class_list_counting = sorted(
        [l.strip() for l in open(args.class_list_counting, 'r').readlines()])


    ###############
    # Build Model #
    ###############

    species_model = model_factory.get_model(
        args.model_arch, len(args.class_list_species), True)

    counting_model = model_factory.get_model(
        args.model_arch, len(args.class_list_counting), True)


    if isfile(args.checkpoint_species) and isfile(args.checkpoint_counting):
        print("Speices checkpoint found at: %s" % args.checkpoint_species)
        print("Counting checkpoint found at: %s" % args.checkpoint_counting)

        if cuda:
            checkpoint_species = torch.load(args.checkpoint_species)
            checkpoint_counting = torch.load(args.checkpoint_counting)
            state_dict_species = checkpoint_species['state_dict']
            state_dict_counting = checkpoint_counting['state_dict']
        else:
            checkpoint_species = torch.load(args.checkpoint_species,
                                    map_location=lambda storage, loc: storage)
            checkpoint_counting = torch.load(args.checkpoint_counting,
                                    map_location=lambda storage, loc: storage)
            state_dict_species = remove_data_parallel(checkpoint_species['state_dict'])
            state_dict_counting = remove_data_parallel(checkpoint_counting['state_dict'])

        species_model.load_state_dict(state_dict_species)
        counting_model.load_state_dict(state_dict_counting)
        print('Successfully loaded checkpoints')
    else:
        print('No checkpoint found at: %s' % args.checkpoint)


    if cuda:
        if torch.cuda.device_count() > 1:
            print("Loading models on %i cuda devices" % torch.cuda.device_count())
            species_model = torch.nn.DataParallel(species_model)
            counting_model = torch.nn.DataParallel(counting_model)
        species_model.cuda()
        counting_model.cuda()
    else:
        species_model.cpu()
        counting_model.cpu()

    species_model.train(False)
    counting_model.train(False)


    ################
    # Analyze Data #
    ################

    for item in os.listdir(args.inference_dir):
        print('Processing: %s' % item)

        item_name, ext = splitext(item)
        item_path = join(args.inference_dir, item)

        results_file = open(join(args.results_dir, item_name + '.csv'), 'w')

        if ext == '' and not 'DS_Store':
            results = process_directory(item_path)

            frame_count = 0
            for species, count in results:
                results_file.write('%i,%s,%s\n' % (frame_count, species, count))
                frame_count += 1

        elif ext.lower() in VIDEO_EXTENSIONS:
            tmp_proc_dir = join(args.inference_dir, 'tmp_proc_dir')
            os.mkdir(tmp_proc_dir)
            results = process_video(item_path, tmp_proc_dir)
            shutil.rmtree(tmp_proc_dir)

            frame_count = 0
            for species, count in results:
                for _ in range(0, args.every_nth_frame):
                    results_file.write(
                        '%i,%s,%s\n' % (frame_count, species, count))
                    frame_count += 1

        elif ext.lower() in IMAGE_EXTENSIONS:
            species, count = process_image(item_path)
            results_file.write('%s,%s' % (species, count))

        else:
            print('%s is not a recognized file type, skipping...' % item)
            continue

        results_file.close()


def process_image(image_path):
    image_name, ext = splitext(image_path)

    if not ext.lower() in IMAGE_EXTENSIONS:
        print('%s is not a recognized image type, skipping...' % image_path)
        return

    image = load_image(image_path)
    image = resize_image(image, args.image_size)
    input = transforms.ToTensor()(image).unsqueeze(0)

    # Get species
    species_output = species_model(input)
    _, idx = torch.max(species_output, dim=1)
    idx = idx.item()
    species = args.class_list_species[idx]

    # Get animal count
    output = counting_model(input)
    _, idx = torch.max(output, dim=1)
    idx = idx.item()
    count = args.class_list_counting[idx]

    return species, count


def process_directory(dir_path):
    print('Processing frames...')
    results = []
    files = os.listdir(dir_path)
    pbar = tqdm(total=len(files))
    for f in files:
        f_path = join(dir_path, f)
        result = process_image(f_path)
        results.append(result)
        pbar.update()
    return results


def process_video(video_path, tmp_proc_dir):
    print('Extracting frames from video...')

    _, ext = splitext(video_path)
    if not ext.lower() in VIDEO_EXTENSIONS:
        print('%s is not a recognized video type, skipping...' % path)
        return

    frame_count = 0
    video_cap = cv2.VideoCapture(video_path)
    while video_cap.isOpened():
        isvalid, frame = video_cap.read()
        if not (frame_count % args.every_nth_frame == 0):
            frame_count += 1
            continue
        if isvalid:
            frame_path = join(tmp_proc_dir, 'frame_%i.png' % frame_count)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        else:
            break
    video_cap.release()

    return process_directory(tmp_proc_dir)


def load_image(path):
    with open(path, 'rb') as f:
        image = Image.open(f)
        return image.convert('RGB')


def resize_image(image, size, dim='width'):
    if dim == 'height':
        scale_percent = args.image_size / float(image.size[1])
        new_width = int(float(image.size[0]) * float(scale_percent))
        image = image.resize((new_width, args.image_size), Image.ANTIALIAS)
    elif dim == 'width':
        scale_percent = args.image_size / float(image.size[0])
        new_height = int(float(image.size[1]) * float(scale_percent))
        image = image.resize((args.image_size, new_height), Image.ANTIALIAS)
    return image


def remove_data_parallel(state_dict):
    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        # remove 'module.' of dataparallel
        if name.startswith('module.'): name = name[7:]
        if name.startswith('0.'): name = name[2:]
        new_state_dict[name] = param
    return new_state_dict


if __name__ == '__main__':
    main()
