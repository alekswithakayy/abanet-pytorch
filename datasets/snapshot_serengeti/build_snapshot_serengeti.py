"""
This script will build the downloaded Snapshot Serengeti images as a PyTorch
ImageFolder dataset using multiple threads.


Usage:

$ python build_snapshot_serengeti.py /input/dir /output/dir [--args]

"""

import os
import argparse
import pandas as pd

from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


####################
# Input Parameters #
####################

parser = argparse.ArgumentParser()

parser.add_argument('input_dir',
                    help='Directory where snapshot serengeti data is located')

parser.add_argument('output_dir',
                    help='Directory to save built datasets')

parser.add_argument('--image_size',
                    default=None,
                    type=int,
                    help='Base image size')

parser.add_argument('--num_threads',
                    default=8,
                    type=int,
                    help='Number of worker threads for processing images')

args = parser.parse_args()


##################
# Initialization #
##################

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

species_list_file = os.path.join(args.input_dir, 'species_list.txt')
species_list = [l.strip() for l in open(species_list_file, 'r').readlines()]


####################
# Helper Functions #
####################

def process_image(input_file, output_file, pbar):
    if os.path.isfile(output_file): return
    if args.image_size:
        img = Image.open(input_file)
        if img.size[0] > img.size[1]:
            scale_percent = args.image_size / float(img.size[1])
            new_width = int(float(img.size[0]) * float(scale_percent))
            img = img.resize((new_width, args.image_size), Image.ANTIALIAS)
        else:
            scale_percent = args.image_size / float(img.size[0])
            new_height = int(float(img.size[1]) * float(scale_percent))
            img = img.resize((args.image_size, new_height), Image.ANTIALIAS)
        img = img.convert('RGB')
        img.save(output_file, format='JPEG')
    else:
        os.rename(input_file, output_file)
    pbar.update()

def process_images(img_ids, input_dir, output_dir, pbar):
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        for id in img_ids:
            img_file = id + '.jpg'
            input_file = os.path.join(input_dir, img_file)
            output_file = os.path.join(output_dir, img_file)
            executor.submit(process_image, input_file, output_file, pbar)


####################
# Build Train Data #
####################

print('Building training dataset...')

train_labels_file = os.path.join(args.input_dir, 'train_labels.csv')
train_labels = pd.read_csv(train_labels_file)

train_images_dir = os.path.join(args.input_dir, 'train_images')
train_dataset_dir = os.path.join(args.output_dir, 'train')
if not os.path.isdir(train_dataset_dir): os.mkdir(train_dataset_dir)

print('Processing %d train images...' % len(train_labels.index))
pbar = tqdm(total=len(train_labels.index))
for species in species_list:
    img_indices = train_labels[train_labels.Species == species].index
    img_ids = train_labels.loc[img_indices, 'CaptureEventID'].tolist()
    if len(img_ids):
        species_dir = os.path.join(train_dataset_dir, species)
        if not os.path.isdir(species_dir): os.mkdir(species_dir)
        process_images(img_ids, train_images_dir, species_dir, pbar)
pbar.close()


#########################
# Build Validation Data #
#########################

print('Building validation dataset...')

val_labels_file = os.path.join(args.input_dir, 'val_labels.csv')
val_labels = pd.read_csv(val_labels_file)

val_images_dir = os.path.join(args.input_dir, 'val_images')
val_dataset_dir = os.path.join(args.output_dir, 'val')
if not os.path.isdir(val_dataset_dir): os.mkdir(val_dataset_dir)

print('Processing %d validation images...' % len(val_labels.index))
pbar = tqdm(total=len(val_labels.index))
for species in species_list:
    img_indices = val_labels[val_labels.Species == species].index
    img_ids = val_labels.loc[img_indices, 'CaptureEventID'].tolist()
    if len(img_ids):
        species_dir = os.path.join(val_dataset_dir, species)
        if not os.path.isdir(species_dir): os.mkdir(species_dir)
        process_images(img_ids, val_images_dir, species_dir, pbar)
pbar.close()

print('Finished building dataset.')
