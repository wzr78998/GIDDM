
from __future__ import print_function
import argparse


parser = argparse.ArgumentParser(description='PyTorch code for GIDDM/cGIDDM')

# Data Level
parser.add_argument('--dataset', type=str, default='Houston',
                    help='Houston, pavia')
parser.add_argument('--source_domain', type=str, default='Houston13',
                    help='Houston13, paviaC, W ')
parser.add_argument('--target_domain', type=str, default='Houston18',
                    help='Houston18, paviaU, W ')
parser.add_argument('--seed', type=int, default=3824755, metavar='S',
                    help='7467915，3824755，7467915，1854720，5761742')

## Model Level
parser.add_argument('--model', type=str, default='GIDDM',
                    help='')
parser.add_argument('--net', type=str, default='DCRN_02', metavar='B',
                    help='resnet50, Cc2Net87,DCRN_02')
parser.add_argument('--bottle_neck_dim', type=int, default=300, metavar='B',
                    help='bottle_neck_dim for the classifier network.')
parser.add_argument('--bottle_neck_dim2', type=int, default=5000, metavar='B',
                    help='bottle_neck_dim for the classifier network.')
parser.add_argument('--fea_dim', type=int, default=102, metavar='B',
                    help='fea_dim.')
parser.add_argument('--patches', type=int, default=1, metavar='B',
                    help='patches.')

## Iteration Level
parser.add_argument('--warmup_iter', type=int, default=1000, metavar='S',
                    help='warmup iteration for posterior inference')
parser.add_argument('--training_iter', type=int, default=100, metavar='S',
                    help='training_iter')
parser.add_argument('--update_term', type=int, default=1, metavar='S',
                    help='update term for posterior inference')
## Loss Level
parser.add_argument('--threshold', type=float, default=0.85, metavar='fixmatch',
                    help='threshold for fixmatch')
parser.add_argument('--ls_eps', type=float, default=0.1, metavar='LR',
                    help='label smoothing for classification')

## Optimization Level
parser.add_argument('--update_freq_D', type=int, default=10, metavar='S',
                    help='freq for D in optimization.')
parser.add_argument('--update_freq_G', type=int, default=1, metavar='S',
                    help='freq for G in optimization.')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--scheduler', type=str, default='cos',
                    help='learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                    help='label smoothing for classification')
parser.add_argument('--e_lr', type=float, default=0.002, metavar='LR',
                    help='label smoothing for classification')
parser.add_argument('--g_lr', type=float, default=0.1, metavar='LR',
                    help='label smoothing for classification')

parser.add_argument('--opt_clip', type=float, default=0.1, metavar='LR',
                    help='label smoothing for classification')
## etc:
parser.add_argument('--exp_code', type=str, default='Test', metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--result_dir', type=str, default='results', metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--set_gpu', type=int, default=0,
                    help='gpu setting 0 or 1')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable cuda')
##weight
parser.add_argument('--w_ctu', type=float, default=0.24,
                    help='disable cuda')
parser.add_argument('--w_ctk', type=float, default=0.5,
                    help='disable cuda')
parser.add_argument('--w_l', type=float, default=0.2,
                    help='disable cuda')
parser.add_argument('--w_kl', type=float, default=0.1,
                    help='disable cuda')
parser.add_argument('--dim1', type=int, default=260, help='4,5,6,7,8.')
parser.add_argument('--back_num', type=int, default=1, help='4,5,6,7,8.')
parser.add_argument('--dim2', type=int, default=36, help='4,5,6,7,8.')
parser.add_argument('--dim3', type=int, default=352, help='4,5,6,7,8.')
parser.add_argument('--dim4', type=int, default=140, help='4,5,6,7,8.')
parser.add_argument('--dim5', type=int, default=140, help='4,5,6,7,8.')
parser.add_argument('--dim6', type=int, default=239, help='4,5,6,7,8.')
parser.add_argument('--dim7', type=int, default=79, help='4,5,6,7,8.')
parser.add_argument('--dim8', type=int, default=459, help='4,5,6,7,8.')

try:
    args = parser.parse_args()
except:
    args, _ = parser.parse_known_args()