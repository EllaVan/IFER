import os
import sys
import runner0117
import argparse
import torch
import pytz
import datetime
import random


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu choice')
    parser.add_argument('--manualSeed', type=int, default=20, help='manual seed')
    parser.add_argument('--continue_from', type=int, default=0, help='save interval')
    parser.add_argument('--dataset', type=str, default="AffectNet")
    parser.add_argument('--log_dir', type=str, default="save/AffectNet/try0118", help='log_dir')
    parser.add_argument('--continue_file', type=str,
                        default=None,
                        help='net state file')
    parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=10, help='number of epochs to save')
    parser.add_argument('--Deconv_step', type=int, default=5, help='number of epochs to train Deconv')
    parser.add_argument('--GRAY', default=True, help='generate gray image or not')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--size_mot', type=int, default=512, help='dimension of motion')
    parser.add_argument('--size_att', type=int, default=512, help='dimension of attribute')
    parser.add_argument('--size_res', type=int, default=512, help='dimension of image feature')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[512, 1024])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 512])

    parser.add_argument("--encoder_use_mot", default=False, help="Encoder use motion as input")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.empty_cache()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)

    global device
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print('using gpu:', args.gpu)

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    setup_seed(args.manualSeed)

    print('---- START RUNNING ----')
    runner0117.train(device, args)

