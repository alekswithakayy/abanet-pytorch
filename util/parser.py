import argparse
import configparser


def collect_args():
    # Parse cli args
    cli_parser = argparse.ArgumentParser()
    _, cli_args = cli_parser.parse_known_args()

    # Parse config file args
    file_parser = configparser.ConfigParser()
    # First cli arg is config file path
    file_parser.read(cli_args[0])
    file_args = file_parser.items('CONFIG ARGS')

    # Merge cli args and file args with default args
    args = merge_args(cli_args, file_args)

    return args


def merge_args(cli_args, file_args):

    parser = argparse.ArgumentParser()

    #########
    # Steps #
    #########

    parser.add_argument('--train_cam',
                        default=False,
                        type=str2bool,
                        help='Train the backbone model on classification task')

    parser.add_argument('--make_cam',
                        default=False,
                        type=str2bool,
                        help='Create cams for images in irn_dataset')

    parser.add_argument('--make_ir_label',
                        default=False,
                        type=str2bool,
                        help='Create ir labels for images in irn_dataset')

    parser.add_argument('--train_irn',
                        default=False,
                        type=str2bool,
                        help='Train the irn model on pixel wise affinity task')

    parser.add_argument('--make_ins_seg_labels',
                        default=False,
                        type=str2bool,
                        help='Create instance segmentation labels with irn')

    parser.add_argument('--make_sem_seg_labels',
                        default=False,
                        type=str2bool,
                        help='Create semantic segmentation labes with irn')
                        

    ###############
    # System args #
    ###############

    parser.add_argument('--num_threads',
                        default=4,
                        type=int,
                        help='Number of data preprocessing threads.')

    parser.add_argument('--num_gpus',
                        default=1,
                        type=int,
                        help='Number of GPUs.')

    ##############
    # Model args #
    ##############

    parser.add_argument('--architecture',
                        default='',
                        type=str,
                        help='Name of model architecture. See ./models')

    parser.add_argument('--pretrained',
                        default=False,
                        type=str2bool,
                        help='Use pre-trained model.')

    parser.add_argument('--checkpoint',
                        default=None,
                        type=str,
                        help='Path to checkpoint aka state dictionary.')

    parser.add_argument('--num_classes',
                        default=None,
                        type=int,
                        help='Number of classes')

    ################
    # Dataset args #
    ################

    parser.add_argument('--dataset',
                        default='dataset_folder',
                        type=str,
                        help='Name of pytorch dataset to use.'
                             'See dataset_factory.py')

    parser.add_argument('--dataset_dir',
                        default=None,
                        help='Directory containing dataset')

    parser.add_argument('--image_size',
                        default='',
                        type=str2list,
                        help='Size of transformed image ie height,width')

    parser.add_argument('--batch_size',
                        default=None,
                        type=int,
                        help='Mini batch size.')

    ##############
    # Train args #
    ##############

    parser.add_argument('--epochs',
                        default=None,
                        type=int,
                        help='Number of epochs to run.')

    parser.add_argument('--start_epoch',
                        default=-1,
                        type=int,
                        help='Epoch to start training at (affects learning rate)')

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
                        type=str2list,
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

    parser.add_argument('--model_save_dir',
                        default=None,
                        help='Directory to output model checkpoints')


    #############
    # Test args #
    #############

    parser.add_argument('--test_print_freq',
                        default=10,
                        type=int,
                        help='Print every n iterations.')

    ##############
    # Infer args #
    ##############

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

    parser.add_argument('--infer_crop',
                        default='',
                        type=str2list,
                        help='Crops rectangular region from input image. The '
                        'box is a 4-tuple defining the left, upper, right, and '
                        'lower pixel coordinate.')

    parser.add_argument('--six_crop',
                        default=False,
                        type=str2bool,
                        help='Crop input image/video six ways - the four '
                        'corners, center and full image')

    parser.add_argument('--infer_image_size',
                        default=None,
                        type=str2list,
                        help='Size of transformed image ie width,heights')

    parser.add_argument('--infer_batch_size',
                        default=None,
                        type=int,
                        help='Mini batch size.')

    parser.add_argument('--visualize_results',
                        default=False,
                        type=str2bool,
                        help='Create a visual representation of processed '
                        'video/image results')

    # File args override default args
    parser.set_defaults(**dict(file_args))

    # cli args override all args
    args, _ = parser.parse_known_args(cli_args)

    return args


def str2list(value):
    return [int(s) for s in value.split(',')]

def str2bool(value):
    return value.strip().lower() == 'true'
