import argparse

def str2bool(value):
    return value.strip().lower() == 'true'

def add_system_args(parser):
    parser.add_argument('--train',
                        default=False,
                        type=str2bool,
                        help='Train the model')
    parser.add_argument('--test',
                        default=False,
                        type=str2bool,
                        help='Test the model')
    parser.add_argument('--infer',
                        default=False,
                        type=str2bool,
                        help='Perform inference with the model')
    return parser


def add_dataset_args(parser):
    parser.add_argument('--dataset',
                        default='image_folder',
                        type=str,
                        help='Name of pytorch dataset to use')
    parser.add_argument('--dataset_dir',
                        default=None,
                        help='Directory containing dataset')
    parser.add_argument('--image_size',
                        default=None,
                        type=int,
                        help='Size of transformed image ie size x size')
    parser.add_argument('--num_threads',
                        default=4,
                        type=int,
                        help='Number of data loading threads.')
    parser.add_argument('--batch_size',
                        default=None,
                        type=int,
                        help='Mini batch size.')
    return parser


def add_train_args(parser):
    parser.add_argument('--epochs',
                        default=None,
                        type=int,
                        help='Number of epochs to run.')
    parser.add_argument('--start_epoch',
                        default=-1,
                        type=int,
                        help='Epoch to start training at (effects learning rate)')
    parser.add_argument('--print_freq',
                        default=10,
                        type=int,
                        help='Print every n iterations.')
    parser.add_argument('--criterion',
                        default='CrossEntropyLoss',
                        type=str,
                        help='Loss function.')
    parser.add_argument('--optimizer',
                        default='SGD',
                        type=str,
                        help='Network optimizer.')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_decay',
                        default=0.1,
                        type=float,
                        help='Amount to multiply lr every lr_decay_epochs.')
    parser.add_argument('--lr_decay_epochs',
                        default=1,
                        type=int,
                        help='Learning rate decays every lr_decay_epochs.')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='Momentum.')
    parser.add_argument('--weight_decay',
                        default=1e-4,
                        type=float,
                        help='Weight decay.')
    parser.add_argument('--params_to_train',
                        default=None,
                        help='Regex expression to select trainable params '
                        'based on their name. See examples below. Use '
                        'model.named_parameters() to see param names. '
                        'Matching params: '
                        'params_to_train = ^.*(layer4|fc).*$ '
                        'Excluding params: '
                        'params_to_train = ^((?!ResNet\/(conv1|bn1)).)*$')
    parser.add_argument('--params_to_randomize',
                        default=None,
                        help='Regex expression to select params to randomize'
                        'based on their name. See examples in params_to_train.')
    parser.add_argument('--models_dir',
                        default=None,
                        help='Directory to output model checkpoints')
    return parser


def add_test_args(parser):
    parser.add_argument('--criterion',
                        default='CrossEntropyLoss',
                        type=str,
                        help='Loss function.')
    parser.add_argument('--print_freq',
                        default=10,
                        type=int,
                        help='Print every n iterations.')
    return parser


def add_infer_args(parser):
    parser.add_argument('--inference_dir',
                        default='/data',
                        help='Directory containing images/videos to be inferenced.')
    parser.add_argument('--results_dir',
                        default='/data',
                        help='Directory where results will be saved.')
    parser.add_argument('--class_list',
                        default='',
                        type=str,
                        help='Path to list of classes.')
    parser.add_argument('--every_nth_frame',
                        default=30,
                        type=int,
                        help='Process every nth frame in a video.')
    parser.add_argument('--image_size',
                        default=None,
                        type=int,
                        help='Size of transformed image ie size x size')
    parser.add_argument('--prm',
                        default=False,
                        type=str2bool,
                        help='Perform peak response mapping')
    return parser


def add_model_args(parser):
    parser.add_argument('--architecture',
                        default='',
                        type=str,
                        help='Name of model architecture')
    parser.add_argument('--pretrained',
                        default=False,
                        type=str2bool,
                        help='Use pre-trained model.')
    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        help='Path to latest checkpoint.')
    parser.add_argument('--num_classes',
                        default=None,
                        type=int,
                        help='Number of classes')
    parser.add_argument('--return_peaks',
                        default=False,
                        type=str2bool,
                        help='Return peaks from class response maps')
    return parser


section_parser_map = {
    'SYSTEM': add_system_args,
    'DATASET': add_dataset_args,
    'TRAIN': add_train_args,
    'TEST': add_test_args,
    'INFER': add_infer_args,
    'MODEL': add_model_args
}

def get_section_parser(section):
    parser = argparse.ArgumentParser()
    return section_parser_map[section](parser)
