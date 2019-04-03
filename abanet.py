import argparse
import configparser

import train
import test
import infer

from util.parser import get_section_parser

# Parse command line arguments as meta_args
parser = argparse.ArgumentParser()
parser.add_argument('--config_filepath', '-c',
                    required=True,
                    type=str,
                    help='Path to system configuration file.')
meta_args, args_from_cmdline = parser.parse_known_args()

# Parse configuration file arguments
config = configparser.ConfigParser()
config.read([meta_args.config_filepath])

def get_section_args(section_name):
    parser = get_section_parser(section_name)
    # Set args from config file
    parser.set_defaults(**dict(config.items(section_name)))
    # Command line args override config file
    args, _ = parser.parse_known_args(args_from_cmdline)
    return args

system_args = get_section_args('SYSTEM')
dataset_args = get_section_args('DATASET')
model_args = get_section_args('MODEL')

if system_args.train:
    train_args = get_section_args('TRAIN')
    train.run(train_args, dataset_args, model_args)

if system_args.test:
    test_args = get_section_args('TEST')
    test.run(test_args, dataset_args, model_args)

if system_args.infer:
    infer_args = get_section_args('INFER')
    infer.run(infer_args, model_args)
