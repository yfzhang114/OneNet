from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pdb
import numpy as np
from einops import rearrange
from collections import OrderedDict
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

import os
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from utils.augmentations import Augmenter
import math

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss

class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.input_channels_dim = args.enc_in
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.augmenter = None
        self.aug = args.aug
        self.MMD_loss = MMD_loss(kernel_mul=2.0, kernel_num=5)

    
    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz;
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        train_x, l_tr = train_data.data_x, train_data.data_x.shape[0]
        val_x, l_val = vali_data.data_x, vali_data.data_x.shape[0]
        test_x, l_te = test_data.data_x, test_data.data_x.shape[0]
        repeat = 10
        
        train_x_ = torch.from_numpy(train_x)
        test_x_ = torch.from_numpy(test_x)
        idx_train = np.arange(l_tr)
        idx_te = np.arange(l_te)
        l_idx = min(l_tr, l_te)
        mmd = 0
        for i in range(repeat):
            idx_tr = np.random.choice(idx_train, l_idx, replace=False)
            idx_te = np.random.choice(idx_te, l_idx, replace=False)
            
            mmd += self.MMD_loss(train_x_[idx_tr], test_x_[idx_te])
        print('mmd distance is', mmd / repeat)
        exit()
        # step=5
        # channels = 4
        # name = f"{self.args.data}_channel{channels}_step{step}"
        # x = np.arange(train_x.shape[0])
        # x_val = np.arange(train_x.shape[0], train_x.shape[0]+val_x.shape[0])
        # x_te = np.arange(train_x.shape[0]+val_x.shape[0], train_x.shape[0]+test_x.shape[0]+val_x.shape[0])
        # style_dict = {
        #     '0':dict(linestyle='-', marker='o',markersize=0.1,color='#dd7e6b'),
        #     '1':dict(linestyle='-',marker='o',markersize=0.1,color='#b6d7a8'),
        #     '2':dict(linestyle='-',marker='o',markersize=0.1,color='#f9cb9c'),
        #     '3':dict(linestyle='-',marker='o',markersize=0.1,color='#a4c2f4'), 
        #     '4':dict(linestyle='-',marker='o',markersize=0.1,color='#b4a7d6')
        # }
        # style_dict_te = {
        #     '0':dict(linestyle='--', marker='+',markersize=0.1,color='#dd7e6b'),
        #     '1':dict(linestyle='--',marker='+',markersize=0.1,color='#b6d7a8'),
        #     '2':dict(linestyle='--',marker='+',markersize=0.1,color='#f9cb9c'),
        #     '3':dict(linestyle='--',marker='+',markersize=0.1,color='#a4c2f4'), 
        #     '4':dict(linestyle='--',marker='+',markersize=0.1,color='#b4a7d6')
        # }

        # for i in range(channels-1, channels):
        #     tr_x, te_x, v_x = train_x[:,i], test_x[:,i], val_x[:, i]
        #     plt.plot(x[::step], tr_x[::step], **style_dict[str(i%5)], label=f'Train {i}')
        #     plt.plot(x_val[::step], v_x[::step], **style_dict[str((i+1)%5)], label=f'Val {i}')
        #     plt.plot(x_te[::step], te_x[::step], **style_dict_te[str((i+2)%5)], label=f'Test {i}')
        # plt.ylabel('Value')#, fontdict=font_y)
        # plt.xlabel('Time step')#, fontdict=font_y)
        # #plt.xlabel('Treatment Selection Bias', fontdict=font_y)
        # plt.legend()
        # plt.savefig(f'imgs/{name}.jpg')
        # exit()
        