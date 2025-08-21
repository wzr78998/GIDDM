from __future__ import print_function

import nni
import torch

from data_loader.get_loader import get_dataset_information, get_loader
import random
from utils import utils as utils
from models.basenet import *
import datetime
import numpy as np
import time
import datetime
import warnings
from data_loader.base import UDADataset
import os
import sys
import scipy.io as sio
warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader
import loaddata
def main(args):
    t1 = time.time()
    sum_str = ''
    args_list = [str(arg) for arg in vars(args)]


    args_list.sort()
    for arg in args_list:
        sum_str += '{:>20} : {:<20} \n'.format(arg, getattr(args, arg))

    # tuner_params = nni.get_next_parameter()
    utils.setGPU(args.set_gpu)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.device = torch.device("cuda:%s" % (0))
    print(torch.backends.cudnn.version())
    # torch.set_num_threads(1)

    source_data, target_data, evaluation_data, num_class = get_dataset_information(args.dataset, args.source_domain, args.target_domain)
    # train_transforms, test_transforms = utils.bring_data_transformation()
    #
    # src_dset = UDADataset(source_data, source_data, num_class, train_transforms, test_transforms, is_target=False, batch_size=args.batch_size)
    # src_train_dset, _, _ = src_dset.get_dsets()
    # target_dset = UDADataset(target_data, target_data, num_class, train_transforms, test_transforms, is_target=True, batch_size=args.batch_size)
    # target_dset.get_dsets()


    ###########################################################################################
    data_name = 'Houston'

    if data_name=='Houston':
        data_path_s = 'data/Houston/Houston13.mat'
        label_path_s = 'data/Houston/Houston13_7gt.mat'
        data_path_t = 'data/Houston/Houston18.mat'
        label_path_t = 'data/Houston/Houston18_7gt.mat'
        data_s, label_s = loaddata.load_data_houston(data_path_s, label_path_s)
        # 将第6类和第7类的标签置0
        label_s[label_s == 6] = 0
        label_s[label_s == 7] = 0
        counts = np.bincount(label_s.ravel(), minlength=6)
        print(counts)
        data_t, label_t = loaddata.load_data_houston(data_path_t, label_path_t)
        # 将第6类的标签置0
        label_t[label_t == 6] = 0
        # 将标签为7的改为6
        label_t[label_t == 7] = 6
        counts1 = np.bincount(label_t.ravel(), minlength=7)
        print(counts1)
    if data_name=='Pavia':
        data_path_s = './data/Pavia/pavia.mat'
        label_path_s = './data/Pavia/pavia_gt_7.mat'
        data_path_t = './data/Pavia/paviaU.mat'
        label_path_t = './data/Pavia/paviaU_gt_7.mat'
        data_s, label_s = loaddata.load_data_pavia(data_path_s, label_path_s)
        # 将第6类和第7类的标签置0

        # ind=np.where(label_s==0)
        # rand=random.sample(range(0,len(ind[0])),args.back_num)
        # label_s[ind[0][rand],ind[1][rand]]=8
        label_s[label_s == 6] = 0
        label_s[label_s == 7] = 0
        # label_s[label_s == 8] = 6


        counts = np.bincount(label_s.ravel(), minlength=6)
        print(counts)
        data_t, label_t = loaddata.load_data_pavia(data_path_t, label_path_t)
        # 将标签为6=7+6
        label_t[label_t == 7] = 6

        counts1 = np.bincount(label_t.ravel(), minlength=7)
        print(counts1)
    if data_name=='KSC':

        sdata = sio.loadmat('./data/KSC/ksc1_std.mat')
        for v in sdata.values():
            if isinstance(v, np.ndarray):  # isinstance() 函数来判断一个对象是否是一个已知的类型
                sdata = v
        data_s = torch.Tensor(sdata[:, 1:])
        slabel = sdata[:, 0] - 1  # 标签从0开始
        label_s = torch.Tensor(np.squeeze(slabel)).type(torch.LongTensor)  # np.squeeze从数组的形状中删除单维条目，即把shape中为1的维度去掉
        tdata = sio.loadmat('./data/KSC/ksc3_std.mat')
        for v in tdata.values():
            if isinstance(v, np.ndarray):
                tdata = v
        data_t = torch.Tensor(tdata[:, 1:])
        tlabel = tdata[:, 0] - 1
        label_t = torch.Tensor(np.squeeze(tlabel)).type(torch.LongTensor)
        fea_dim = data_s.shape[1]
        data_s = data_s[~(label_s == 7) & ~(label_s == 8) & ~(label_s == 9)]
        label_s=torch.where(label_s!=7,label_s,torch.tensor(float('nan')))
        label_s = torch.where(label_s != 8, label_s, torch.tensor(float('nan')))
        label_s = torch.where(label_s != 9, label_s, torch.tensor(float('nan')))
        label_s=label_s[~torch.isnan(label_s)].long()

        counts = np.bincount(label_s.ravel(), minlength=7)
        print(counts)
        label_t[label_t == 9] = 7
        label_t[label_t == 8] = 7
        label_t[label_t == 9] = 0
        label_t[label_t == 8] = 0
        counts1 = np.bincount(label_t.ravel(), minlength=8)
        num_class=len(counts1)
        print(counts1)


    _,trainX, trainY,_,_,_,_,_= loaddata.get_all_data(data_s, label_s, int((args.patches-1)/2))
    testID, testX, testY, G, RandPerm, Row, Column,indices = loaddata.get_all_data(data_t, label_t, int((args.patches-1)/2))
    # trainX, trainY=loaddata.get_all_data_ksc(data_s, label_s)
    # testX, testY=loaddata.get_all_data_ksc(data_t, label_t)

    train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))
    tar_label=[RandPerm, Row, Column,G]
    train_loader_s = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_loader_t = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)
###
    dataset_info = '%s_%s_%s' % (args.dataset, args.source_domain, args.target_domain)
    save_info = '%s#%s#%s#%s#%s' % (args.exp_code, args.model, args.net, dataset_info, args.seed)

    result_model_dir = utils.get_save_dir(args, save_info)

    logger = utils.bring_logger(result_model_dir)
    args.logger = logger
    args.logger.info(sum_str)
    args.logger.info('=' * 30)


    from models.model_HSI_HS import GIDDM
    model = GIDDM(args, num_class, train_loader_s, train_loader_t,tar_label,test_loader)

    #
    model.train_init()
    model.test(0)
    model.build_model()
    model.train()

if __name__ == '__main__':
    from config_HSI import args


    hyperparams = vars(args)  # 超参
    optimized_params = nni.get_next_parameter()
    hyperparams.update(optimized_params)
    # print(hyperparams)
    # print('gpu:',torch.cuda.current_device())
    main(args)
