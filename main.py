import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
# import torch.utils.tensorboard as tb
import time

# from runners.diffusion import Diffusion
from guided_diffusion.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
t1 = time.localtime(time.time())
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation"
    )
    parser.add_argument(
        "--path_y",
        type=str,
        required=True,
        help="Path of the test dataset.",
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0., help="sigma_y"
    )
    parser.add_argument(
        "--eta", type=float, default=0.85, help="Eta"
    )    
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Use simplified DDNM, without SVD",
    )    
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--deg_scale", type=float, default=0., help="deg_scale"
    )    
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "-n",
        "--noise_type",
        type=str,
        default="gaussian",
        help="gaussian | 3d_gaussian | poisson | speckle"
    )
    parser.add_argument(
        "--add_noise",
        action="store_true"
    )
    parser.add_argument("--A_type", type=str, default="math",
                         help="A type: math, adapter")
    parser.add_argument("--adap_pth", type=str, default="empty",
                         help="adapter pth file path")
    parser.add_argument('--nums_rb', type=int, default=20,
                         help='number of residual blocks in adpter')
    parser.add_argument('--add_temb', type=bool, default=False,
                         help='if add time embedding in adapter')
    parser.add_argument('--res_adap', type=bool, default=False,
                         help='if add residual in whole scale adapter')    
    parser.add_argument("--noise_hyper", type=str, default="ddnm",
                         help="ddnm or jpeg_ddrm")
    parser.add_argument("--init_x", type=str, default="pure_noise",
                         help="pure_noise, Apy and noisy_Apy")
    parser.add_argument("--mask", type=str, default="mask",
                         help="mask type: facemask, half, draw, mask(ink)")


    parser.add_argument(
        "--time_travel",
        action="store_false",
        help="if cancel time_travel",
    )                


    # train
    parser.add_argument('--save', default='save/-{}-{}-{}-{}-{}-{}'.format(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory')
    parser.add_argument("--lr", type=float, default="1e-5",
                         help="learining rate")
    parser.add_argument("--train_size", type=int, default=1,
                         help='size of the traindata')
    parser.add_argument("--num_trainset", type=int, default=10000,
                         help='number of the traindata')
    parser.add_argument("--num_valset", type=int, default=100,
                         help='number of the valdata')
    parser.add_argument('--epochs', type=int, default=100,
                         help='number of epochs')
    parser.add_argument('--re_epoch', type=int, default=0,
                         help='number of re_epochs for train going on last_train')
    parser.add_argument('--batch_size', type=int, default=1,
                         help='batch size')
    parser.add_argument('--slice_size', type=int, default=1,
                         help='slice size')
    parser.add_argument('--img_size', type=int, default=256,
                         help='image size')
    parser.add_argument('--num_classes', type=int, default=2,
                         help='number of classes')
    parser.add_argument('--noise_steps', type=int, default=1000,
                         help='noise steps')
    parser.add_argument('--num_workers', type=int, default=1,
                         help='num of workers')
    parser.add_argument('--clip_max_norm', type=int, default=1.0,
                         help='gradient clipping max norm')
    parser.add_argument('--is_visualization', default=False, type=bool,
                         help='visualization when test or not')  
    parser.add_argument('--is_test', default=False, type=bool,
                         help='skip train to do test')  
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                       metavar='N', help='print frequency (default: 100)')
    
    # others
    parser.add_argument('--visual_every_sample', default=False, type=bool,
                     help='if visual_every_sample')


    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    res = os.path.join(args.image_folder, "res")
    gt = os.path.join(args.image_folder, "gt")
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    if not os.path.exists(res):
        os.makedirs(res)
    if not os.path.exists(gt):
        os.makedirs(gt)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    try:
        runner = Diffusion(args, config)
        runner.sample(args.simplified)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
