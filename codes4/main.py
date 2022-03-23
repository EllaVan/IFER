import os
import argparse
import torch
import pytz
import datetime
import random

import runner0120
import mot_gen_nN


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cuda:1', help='gpu choice')
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    parser.add_argument('--continue_from', type=int, default=0, help='save interval')
    parser.add_argument('--dataset', type=str, default="AffectNet")
    parser.add_argument('--log_dir', type=str, default="save", help='log_dir')
    parser.add_argument('--continue_file', type=str,
                        default=None,
                        help='net state file')
    parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=10, help='number of epochs to save')
    parser.add_argument('--Deconv_step', type=int, default=5, help='number of epochs to train Deconv')
    parser.add_argument('--GRAY', default=True, help='generate gray image or not')

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--size_mot', type=int, default=512, help='dimension of motion')
    parser.add_argument('--size_att', type=int, default=512, help='dimension of attribute')
    parser.add_argument('--size_res', type=int, default=512, help='dimension of image feature')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[512, 1024])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 512])

    parser.add_argument("--encoder_use_mot", default=False, help="Encoder use motion as input")

    return parser.parse_args()



def mot_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='cuda:1', help='gpu choice')
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='save interval')
    parser.add_argument('--end_epoch', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=20, help='number of epochs to save for')
    parser.add_argument('--dataset', type=str, default="AffectNet")
    parser.add_argument('--log_dir', type=str, default="save", help='log_dir')
    parser.add_argument('--means_file', type=str,
                        default=None,
                        help='net state file')
    parser.add_argument('--start_file', type=str,
                        default=None,
                        help='net state file')

    parser.add_argument('--num_query', type=int, default=2)
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
    mean_args = mean_parse_args()
    mot_args = mot_parse_args()
    torch.cuda.empty_cache()
    global device
    device = torch.device(mean_args.gpu if torch.cuda.is_available() else 'cpu')
    print('using gpu:', mean_args.gpu)

    cur_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    print(cur_time)
    cur_day = str(cur_time).split(' ')
    cur_day = cur_day[0]
    mot_args.log_dir = os.path.join(mean_args.log_dir, mean_args.dataset, cur_day, 'decoder')
    mean_args.log_dir = os.path.join(mean_args.log_dir, mean_args.dataset, cur_day, 'encoder')

    print('---- START %s Gauss Encoder ----' % (mean_args.dataset))
    if mean_args.manualSeed is None:
        mean_args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", mean_args.manualSeed)
    setup_seed(mean_args.manualSeed)

    runner0120.train(device, mean_args)

    print('---- START %s Motion Decoder ----' % (mean_args.dataset))
    print('using means_epoch: ', mean_args.nepoch)
    mot_args.gpu = mean_args.gpu
    mot_args.dataset = mean_args.dataset
    mot_args.means_file = os.path.join(mean_args.log_dir, 'net_state', 'epoch' + str(mean_args.nepoch))
    mot_args.manualSeed = mean_args.manualSeed

    mot_gen_nN.run(device, mot_args)

