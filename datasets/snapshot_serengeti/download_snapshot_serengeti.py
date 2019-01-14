"""
This script will download and clean all data relevant to Snapshot Serengeti
using multiple threads.

Usage:

$ python download_snapshot_serengeti.py /save/to/dir [--args]

"""

import os
import sys
import urllib.request
import argparse
import pandas as pd

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


####################
# Input Parameters #
####################

parser = argparse.ArgumentParser()

def str2bool(value):
    return value.strip().lower() == 'true'

parser.add_argument('output_dir',
                    help='Directory to save all data')

parser.add_argument('--test_run',
                    default=False,
                    type=str2bool,
                    help='Runs in a test mode that only downloads 100 images')

parser.add_argument('--labels_to_keep',
                    default=['CaptureEventID', 'Species', 'Count'],
                    nargs='+',
                    help='Which labels to keep in cleaned labels file')

parser.add_argument('--download_train_imgs',
                    default=True,
                    type=str2bool,
                    help='Download train images')

parser.add_argument('--download_val_imgs',
                    default=True,
                    type=str2bool,
                    help='Download validation images')

parser.add_argument('--include_blank_img_class',
                    default=False,
                    type=str2bool,
                    help='Include blank images in dataset')

parser.add_argument('--num_blank_train_imgs',
                    default=10,
                    type=int,
                    help='Number of blank images to put in training set. \
                          N/A if include_blank_img_class is false')

parser.add_argument('--num_blank_val_imgs',
                    default=10,
                    type=int,
                    help='Number of blank images to put in validation set. \
                          N/A if include_blank_img_class is false')

parser.add_argument('--exclude_multiple_species',
                    default=True,
                    type=str2bool,
                    help='Exclude images containing multiple species')

parser.add_argument('--num_threads',
                    default=8,
                    type=int,
                    help='Number of worker threads for dowloading images')

args = parser.parse_args()


#####################
# Utility Functions #
#####################

# User-agent headers for url requests
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36\
            (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',}

def retrieve_file(file, url, pbar=None):
    if pbar: pbar.update()
    if os.path.isfile(file): return
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request) as response:
        with open(file, 'wb') as f:
            f.write(response.read())


###################
# Load Image URLs #
###################

ALL_IMAGES_FILENAME = 'all_images.csv'
ALL_IMAGES_URL = ('https://datadryad.org/bitstream/handle/10255/dryad.86392/'
                  'all_images.csv?sequence=1')

all_images_file = os.path.join(args.output_dir, ALL_IMAGES_FILENAME)
if not os.path.isfile(all_images_file):
    print('Retrieving %s...' % ALL_IMAGES_FILENAME)
    retrieve_file(all_images_file, ALL_IMAGES_URL)
print('Reading %s...' % ALL_IMAGES_FILENAME)
all_images = pd.read_csv(all_images_file)
print('Done.')

# Map ids to urls
id_to_url = dict(zip(all_images['CaptureEventID'].tolist(), \
                     all_images['URL_Info'].tolist()))


#######################
# Load Consensus Data #
#######################

CONSENSUS_DATA_FILENAME = 'consensus_data.csv'
CONSENSUS_DATA_URL = ('https://datadryad.org/bitstream/handle/10255/'
                      'dryad.86348/consensus_data.csv?sequence=1')

consensus_data_file = os.path.join(args.output_dir, CONSENSUS_DATA_FILENAME)
if not os.path.isfile(consensus_data_file):
    print('Retrieving %s...' % CONSENSUS_DATA_FILENAME)
    retrieve_file(consensus_data_file, CONSENSUS_DATA_URL)
print('Reading %s...' % CONSENSUS_DATA_FILENAME)
all_consensus_data = pd.read_csv(consensus_data_file)
print('Done.')

# Drop images where NumSpecies does not equal 1
if args.exclude_multiple_species:
    to_drop = all_consensus_data[all_consensus_data.NumSpecies != 1].index
    consensus_data = all_consensus_data.drop(to_drop)
# Drop unnecessary labels
consensus_data = consensus_data[args.labels_to_keep]


############################
# Load Expert Labeled Data #
############################

EXPERT_DATA_FILENAME = 'expert_data.csv'
EXPERT_DATA_URL = ('https://datadryad.org/bitstream/handle/10255/dryad.76010/'
                   'gold_standard_data.csv?sequence=1')

expert_data_file = os.path.join(args.output_dir, EXPERT_DATA_FILENAME)
if not os.path.isfile(expert_data_file):
    print('Retrieving %s...' % EXPERT_DATA_FILENAME)
    retrieve_file(expert_data_file, EXPERT_DATA_URL)
print('Reading %s...' % EXPERT_DATA_FILENAME)
all_expert_data = pd.read_csv(expert_data_file)
print('Done.')

# Drop images labelled impossible
to_drop = all_expert_data[all_expert_data.Species=='impossible'].index.tolist()
# Drop rows where NumSpecies does not equal 1
if args.exclude_multiple_species:
    to_drop += all_expert_data[all_expert_data.NumSpecies != 1].index.tolist()
expert_data = all_expert_data.drop(to_drop)
# Replace incorrect species names
expert_data.loc[expert_data.Species == 'rodent', 'Species'] = 'rodents'
# Make sure labels to keep are also in expert dataset
expert_labels_to_keep = [l for l in args.labels_to_keep \
                         if l in expert_data.columns]
# Drop unnecessary labels
expert_data = expert_data[expert_labels_to_keep]


######################
# Write Species List #
######################

print('Writing species list...')
# Creates file listing each unique species name in the dataset
SPECIES_LIST_FILENAME = 'species_list.txt'
species_list_file = os.path.join(args.output_dir, SPECIES_LIST_FILENAME)
species_list = list(consensus_data.Species.unique())
if args.include_blank_img_class:
    species_list.append('blank')
species_list.sort()
with open(species_list_file, 'w') as f:
    f.writelines("\n".join(species_list))
print('Done.')


##################
# Test Mode Init #
##################

if args.test_run:
    consensus_data = consensus_data.loc[:99, args.labels_to_keep]
    expert_data = expert_data.loc[:99, expert_labels_to_keep]


###################
# Map IDs to URLs #
###################

print('Mapping image ids to urls...')
def map_id_to_url(dataframe):
    valid_id_to_url = {}
    to_drop = []
    for idx, id in dataframe.CaptureEventID.iteritems():
        # Exclude imgs w/o url
        if id in id_to_url:
            valid_id_to_url[id] = id_to_url[id]
        else:
            to_drop.append(idx)
    dataframe = dataframe.drop(to_drop)
    return dataframe, valid_id_to_url

consensus_data, train_id_to_url = map_id_to_url(consensus_data)
expert_data, val_id_to_url = map_id_to_url(expert_data)
print('Done.')


#######################
# Select Blank Images #
#######################

# Include first n blank images in training and validation sets
if args.include_blank_img_class:
    print('Selecting blank images...')
    to_add = {'train':[], 'val':[]}
    blank_count = {'train':0, 'val':0}
    # Image is blank if not in one of these datasets
    non_blank_imgs = set(all_consensus_data.CaptureEventID.tolist() +
                         all_expert_data.CaptureEventID.tolist())
    for id in sorted(id_to_url.keys()):
        if id not in non_blank_imgs:
            if blank_count['train'] < args.num_blank_train_imgs:
                train_id_to_url[id] = id_to_url[id]
                # Create entry for blank image
                entry = []
                for label in args.labels_to_keep:
                    if label == 'CaptureEventID': entry.append(id)
                    elif label == 'Species': entry.append('blank')
                    elif label == 'Count': entry.append(0)
                    else: entry.append(0.)
                to_add['train'].append(entry)
                blank_count['train'] += 1
            elif blank_count['val'] < args.num_blank_val_imgs:
                val_id_to_url[id] = id_to_url[id]
                entry = []
                for label in expert_labels_to_keep:
                    if label == 'CaptureEventID': entry.append(id)
                    elif label == 'Species': entry.append('blank')
                    elif label == 'Count': entry.append(0)
                to_add['val'].append(entry)
                blank_count['val'] += 1
            else:
                break

    to_add_df = pd.DataFrame(to_add['train'], columns=args.labels_to_keep)
    consensus_data = pd.concat([consensus_data, to_add_df])

    to_add_df = pd.DataFrame(to_add['val'], columns=expert_labels_to_keep)
    expert_data = pd.concat([expert_data, to_add_df])

    print('Done.')


###################
# Download Images #
###################

# Helper functions

def download_dataset(name, id_to_url, output_dir):
    img_count = len(id_to_url)
    print('Downloading %d %s images...' % (img_count, name))
    pbar = tqdm(total=img_count)
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        for id, url in id_to_url.items():
            url = BASE_URL + url
            file = os.path.join(output_dir, id + '.jpg')
            executor.submit(retrieve_file, file, url, pbar)
    pbar.close()

def validate_image_files(dataframe):
    img_count = dataframe.CaptureEventID.size
    pbar = tqdm(total=img_count)
    to_drop = []
    for idx, id in dataframe.CaptureEventID.iteritems():
        file = os.path.join(output_dir, id + '.jpg')
        # Check that file exists
        if os.path.isfile(file):
            # Check that file is not empty
            if os.path.getsize(file) == 0:
                to_drop.append(idx)
                os.remove(file)
        else:
            to_drop.append(idx)
        pbar.update()
    return dataframe.drop(to_drop)

# Base url for retrieving images
BASE_URL = 'https://snapshotserengeti.s3.msi.umn.edu/'

# Download train data
output_dir = os.path.join(args.output_dir, 'train_images')
if not os.path.isdir(output_dir): os.mkdir(output_dir)
if args.download_train_imgs:
    download_dataset('train', train_id_to_url, output_dir)
print('Validating downloaded train images...')
consensus_data = validate_image_files(consensus_data)
print('%d train images successfully downloaded.' % len(consensus_data.index))

# Download validation data
output_dir = os.path.join(args.output_dir, 'val_images')
if not os.path.isdir(output_dir): os.mkdir(output_dir)
if args.download_val_imgs:
    download_dataset('val', val_id_to_url, output_dir)
print('Validating downloaded validation images...')
expert_data = validate_image_files(expert_data)
print('%d validation images successfully downloaded.' % len(expert_data.index))


#####################
# Write Label Files #
#####################

print('Writing label files...')
# Write clean consensus data to label file
train_labels_file = os.path.join(args.output_dir, 'train_labels.csv')
consensus_data.to_csv(train_labels_file, index=False, header=args.labels_to_keep)
# Write clean expert data to label file
val_labels_file = os.path.join(args.output_dir, 'val_labels.csv')
expert_data.to_csv(val_labels_file, index=False, header=expert_labels_to_keep)

print('Finished downloading data.')
