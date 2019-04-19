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
    parser.add_argument('--output_file',
                        default=None,
                        type=str,
                        help='If specified, redirects output to file')
    return parser


def clean_system_args(args):
    return args


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
    return parser


def clean_model_args(args):
    return args


def add_dataset_args(parser):
    parser.add_argument('--dataset',
                        default='dataset_folder',
                        type=str,
                        help='Name of pytorch dataset to use')
    parser.add_argument('--dataset_dir',
                        default=None,
                        help='Directory containing dataset')
    parser.add_argument('--backgnd_samp_prob',
                        default=0.2,
                        type=float,
                        help='Sampling rate of background class.')
    parser.add_argument('--image_size',
                        default='',
                        type=str,
                        help='Size of transformed image ie height,width')
    parser.add_argument('--num_threads',
                        default=4,
                        type=int,
                        help='Number of data loading threads.')
    parser.add_argument('--batch_size',
                        default=None,
                        type=int,
                        help='Mini batch size.')
    return parser


def clean_dataset_args(args):
    args.image_size = [int(s) for s in args.image_size.split(',')]
    return args


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
    parser.add_argument('--mixed_prec_level',
                        default='O0',
                        type=str,
                        help='Level of mixed precision to use.'
                        'O0 = FP32, O1 = Conservative, O2 = Fast, O3 = FP16'
                        'O1 or O2 recommended. See Nvidia/Apex for details.')
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
                        help='Amount to multiply lr every lr_decay_iters.')
    parser.add_argument('--lr_decay_iters',
                        default='',
                        type=str,
                        help='Learning rate is decayed by lr_decay for each '
                        'iteration in lr_decay_iters.')
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
                        'params_to_train = ^((?!ResNet\/(conv1|bn1)).)*$ ')
    parser.add_argument('--params_to_randomize',
                        default=None,
                        help='Regex expression to select params to randomize'
                        'based on their name. See examples in params_to_train.')
    parser.add_argument('--models_dir',
                        default=None,
                        help='Directory to output model checkpoints')
    return parser


def clean_train_args(args):
    args.lr_decay_iters = [int(s) for s in args.lr_decay_iters.split(',')]
    return args


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


def clean_test_args(args):
    return args


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
    parser.add_argument('--num_threads',
                        default=4,
                        type=int,
                        help='Number of data preprocessing threads.')
    parser.add_argument('--crop',
                        default='',
                        type=str,
                        help='Crops rectangular region from input image. The '
                        'box is a 4-tuple defining the left, upper, right, and '
                        'lower pixel coordinate.')
    parser.add_argument('--six_crop',
                        default=False,
                        type=str2bool,
                        help='Crop input image/video six ways - the four '
                        'corners, center and full image')
    parser.add_argument('--image_size',
                        default=None,
                        type=str,
                        help='Size of transformed image ie width,heights')
    parser.add_argument('--batch_size',
                        default=None,
                        type=int,
                        help='Mini batch size.')
    parser.add_argument('--visualize_results',
                        default=False,
                        type=str2bool,
                        help='Create a visual representation of processed '
                        'video/image results')
    return parser


def clean_infer_args(args):
    args.image_size = [int(s) for s in args.image_size.split(',')]
    if args.crop:
        args.crop = [int(s) for s in args.crop.split(',')]
    return args


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

section_cleaner_map = {
    'SYSTEM': clean_system_args,
    'DATASET': clean_dataset_args,
    'TRAIN': clean_train_args,
    'TEST': clean_test_args,
    'INFER': clean_infer_args,
    'MODEL': clean_model_args
}

def clean_section_args(args, section):
    return section_cleaner_map[section](args)
