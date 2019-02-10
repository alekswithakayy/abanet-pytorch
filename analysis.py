import argparse
import os
import cv2
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as transforms

from PIL import Image
from models import model_factory
from os.path import isfile, join, splitext

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

parser.add_argument('--inference_dir',
                    default='/data',
                    help='Directory containing images/videos to be inferenced.')

def main():

    ##############
    # Initialize #
    ##############

    global args, species_model, counting_model

    args = parser.parse_args()

    # Check if cuda is available
    cuda = torch.cuda.is_available()
    print("Using cuda: %s" % cuda)


    ###############
    # Build Model #
    ###############

    species_model = model_factory.get_model(
        args.model_arch, args.n_classes_species, True)

    counting_model = model_factory.get_model(
        args.model_arch, args.n_classes_counting, True)


    if isfile(args.checkpoint_species) and isfile(args.checkpoint_counting):
        print("Speices checkpoint found at: %s" % args.checkpoint_species)
        print("Counting checkpoint found at: %s" % args.checkpoint_counting)

        if cuda:
            checkpoint_species = torch.load(args.checkpoint_species)
            checkpoint_counting = torch.load(args.checkpoint_counting)
        else:
            checkpoint_species = torch.load(args.checkpoint_species,
                                    map_location=lambda storage, loc: storage)
            checkpoint_species = torch.load(args.checkpoint_counting,
                                    map_location=lambda storage, loc: storage)

        species_model.load_state_dict(checkpoint_species['state_dict'])
        counting_model.load_state_dict(checkpoint_counting['state_dict'])
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
        print('Processing %s' % item)

        _, ext = splitext(item)
        path = join(args.inference_dir, item)

        if ext == '':
            process_directory(path)
        elif ext.lower() in ['.mp4']:
            process_video(path)
        elif ext.lower() in ['.jpeg', '.jpg', '.png']:
            process_image(path)
        else:
            print('%s is not a recognized file type, skipping...' % item)
            continue

def process_image(path):
    image = pil_loader(path)
    image = resize_image(image, args.image_size)
    return species_model(image), counting_model(image)


def pil_loader(path):
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


def extract_frames(video_path, frames_dir, every_nth=10):
    video_cap = cv2.VideoCapture(video_path)
    data_info_txt = open(frames_dir + '.txt', 'w')
    idx = 0
    while video_cap.isOpened():
        isvalid, frame = video_cap.read()
        if not (idx % every_nth == 0):
            idx += 1
            continue
        if isvalid:
            frame_path = os.path.join(frames_dir, 'frame_{}.png'.format(idx))
            #resized_frame = cv2.resize(frame, (256,256))
            #cv2.imwrite(frame_path, resized_frame)
            cv2.imwrite(frame_path, frame)
            data_info_txt.write(frame_path + ',\n')
        else:
            break
        idx += 1
    video_cap.release()
    data_info_txt.close()


if __name__ == '__main__':
    main()
