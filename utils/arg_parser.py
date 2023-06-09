import argparse
import torch

from models.pretrained_models import PretrainedModel

"""
Utility file to parse and print arguments from an argument string.
"""


def get_args(arg_str=None):
    parser = argparse.ArgumentParser()

    # First, define the arguments with their type and default values

    # Original comment: General arguments
    parser.add_argument('--data-path', type=str,
                        default='./data',
                        help="root path of all data")
    parser.add_argument('--checkpoint-path', type=str,
                        default='./checkpoints',
                        help="path checkpoints are stored or loaded")
    parser.add_argument('--log-step', type=int , default=10,
                        help="step size for printing logging information")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="number of threads used by data loader")

    parser.add_argument('--disable-cuda', action='store_true',
                        help="disable the use of CUDA")
    parser.add_argument('--cuda-device', type=int , default=0,
                        help="specify which GPU to use")
    parser.add_argument('--torch-seed', type=int,
                        help="set a torch seed")


    # Original comment: Model parameters
    parser.add_argument('--model', type=str, default='lrcn',
                        help="deep learning model",
                        choices=['lrcn', 'gve', 'sc'])
    parser.add_argument('--dataset', type=str, default='coco',
                        choices=['coco', 'cub'])
    parser.add_argument('--pretrained-model', type=str, default='vgg16',
                        help="[LRCN] name of pretrained model for image features",
                        choices=PretrainedModel.SUPPORTED_MODEL_NAMES)
    parser.add_argument('--layers-to-truncate', type=int, default=1,
                        help="[LRCN] number of final FC layers to be removed from pretrained model")
    parser.add_argument('--sc-ckpt', type=str, default='data/cub/sentence_classifier_ckpt.pth',
                        help="[GVE] path to checkpoint for pretrained sentence classifier")
    parser.add_argument('--weights-ckpt', type=str,
                        help="[GVE] path to checkpoint for pretrained weights")
    parser.add_argument('--loss-lambda', type=float, default=0.2,
                        help="[GVE] weight factor for reinforce loss")

    parser.add_argument('--embedding-size', type=int , default=1000,
                        help='dimension of the word embedding')
    parser.add_argument('--hidden-size', type=int , default=1000,
                        help='dimension of hidden layers')

    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    parser.add_argument('--eval', type=str,
                        help="path of checkpoint to be evaluated")

    # OWN CODE
    # we add an argument that allows to suppress the evaluation in training
    parser.add_argument('--without_eval', action='store_true', default=False, help="suppress evaluation in training")

    # parse the argument string
    if arg_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_str.split())

    # dictionary for arguments {parameter: value}
    arg_vars = vars(args)

    # if there is no checkpoint specified for evaluation, train is True; otherwise, train is false.
    arg_vars["train"] = not args.eval
    # get the checkpoint for evaluation
    arg_vars["eval_ckpt"] = args.eval
    # delete the eval parameter from the dictionary
    del arg_vars["eval"]

    # Original comment: GVE currently does not support pretrained models
    if arg_vars["model"] == "gve":
        arg_vars["pretrained_model"] = None

    # set seed if given. Otherwise, set a default seed
    if args.torch_seed is not None:
        torch.manual_seed(arg_vars["torch_seed"])
    else:
        arg_vars["torch_seed"] = torch.initial_seed()

    return args


def print_args(args):
    """
    Prints the parsed arguments
    """
    space = 30
    print("Arguments:")
    for arg, value in vars(args).items():
        print('{:{space}}{}'.format(arg, value, space=space))
    print()
