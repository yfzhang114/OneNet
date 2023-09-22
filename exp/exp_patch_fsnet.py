from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.fsnet import TSEncoder, GlobalLocalMultiscaleTSEncoder
from models.ts2vec.losses import hierarchical_contrastive_loss
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cumavg
import pdb
import numpy as np
from einops import rearrange
from collections import OrderedDict
import time
import torch
from utils.buffer import Buffer
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

from layers.PatchTST_backbone import PatchTST_backbone_TCN
from layers.PatchTST_layers import series_decomp


class net(nn.Module):
    def __init__(self, configs, device='cuda', max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        self.channel_cross = configs.channel_cross
            
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone_TCN(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, channel_cross=self.channel_cross, tcn_output_dim=configs.tcn_output_dim, tcn_layer=configs.tcn_layer,tcn_hidden=configs.tcn_hidden, **kwargs)
            self.model_res = PatchTST_backbone_TCN(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, channel_cross=self.channel_cross,tcn_output_dim=configs.tcn_output_dim, tcn_layer=configs.tcn_layer,tcn_hidden=configs.tcn_hidden, **kwargs)
        else:
            self.model = PatchTST_backbone_TCN(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, channel_cross=self.channel_cross,tcn_output_dim=configs.tcn_output_dim, tcn_layer=configs.tcn_layer,tcn_hidden=configs.tcn_hidden,**kwargs)
        
        self.to(device)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x

    def store_grad(self):
        if self.decomposition:
            for name, layer in self.model_trend.TCN_backbone.named_modules():    
                if 'PadConv' in type(layer).__name__:
                    #print('{} - {}'.format(name, type(layer).__name__))
                    layer.store_grad()
            for name, layer in self.model_res.TCN_backbone.named_modules():    
                if 'PadConv' in type(layer).__name__:
                    #print('{} - {}'.format(name, type(layer).__name__))
                    layer.store_grad()
        else:
            for name, layer in self.model.TCN_backbone.named_modules():    
                if 'PadConv' in type(layer).__name__:
                    #print('{} - {}'.format(name, type(layer).__name__))
                    layer.store_grad()
class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.input_channels_dim = args.enc_in
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor', 'encoder', 'tcn', 'regressor_tcn']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = net(args, device = self.device)
        self.augmenter = None
        self.aug = args.aug
        self.buffer = Buffer(500, self.device)     
        self.count = 0
        if args.finetune:
            inp_var = 'univar' if args.features == 'S' else 'multivar'
            model_dir = str([path for path in Path(f'/export/home/TS_SSL/ts2vec/training/ts2vec/{args.data}/')
                .rglob(f'forecast_{inp_var}_*')][args.finetune_model_seed])
            state_dict = torch.load(os.path.join(model_dir, 'model.pkl'))
            for name in list(state_dict.keys()):
                if name != 'n_averaged':
                    state_dict[name[len('module.'):]] = state_dict[name]
                del state_dict[name]
            self.model[0].encoder.load_state_dict(state_dict)

    def get_augmenter(self, sample_batched):
    
        seq_len = sample_batched.shape[1]
        num_channel = sample_batched.shape[2]
        cutout_len = math.floor(seq_len / 12)
        if self.input_channels_dim != 1:
            self.augmenter = Augmenter(cutout_length=cutout_len)
        #IF THERE IS ONLY ONE CHANNEL, WE NEED TO MAKE SURE THAT CUTOUT AND CROPOUT APPLIED (i.e. their probs are 1)
        #for extremely long sequences (such as SSC with 3000 time steps)
        #apply the cutout in multiple places, in return, reduce history crop
        elif self.input_channels_dim == 1 and seq_len>1000: 
            self.augmenter = Augmenter(cutout_length=cutout_len, cutout_prob=1, crop_min_history=0.25, crop_prob=1, dropout_prob=0.0)
            #we apply cutout 3 times in a row.
            self.augmenter.augmentations = [self.augmenter.history_cutout, self.augmenter.history_cutout, self.augmenter.history_cutout,
                                            self.augmenter.history_crop, self.augmenter.gaussian_noise, self.augmenter.spatial_dropout]
        #if there is only one channel but not long, we just need to make sure that we don't drop this only channel
        else:
            self.augmenter = Augmenter(cutout_length=cutout_len, dropout_prob=0.0)
    
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

    def _select_optimizer(self):
        # self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # self.opt = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.opt = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss, aug_loss = [], []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                if self.augmenter == None:
                   self.get_augmenter(batch_x)

                self.opt.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                
                loss = criterion(pred, true)
                
                if self.aug > 0:
                    loss_aug = 0
                    for i in range(self.aug):
                        batch_xa, _ = self.augmenter(batch_x.float().to(self.device), torch.ones_like(batch_x).to(self.device))
                        pred, true = self._process_one_batch(train_data, batch_xa, batch_y, batch_x_mark, batch_y_mark)
                        loss_aug += criterion(pred, true)
                    loss_aug /= self.aug
                    loss += self.args.loss_aug * loss_aug
                else:
                    loss_aug = torch.tensor(0)
                
                train_loss.append(loss.item())
                aug_loss.append(loss_aug.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
                self.model.store_grad()
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            aug_loss = np.average(aug_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = 0.
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Aug Loss {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, aug_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='vali')
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        # for param_group in self.opt.param_groups:
        #     param_group['lr'] = self.args.lr_test
        self.model.eval()
        if self.online == 'regressor':
            if self.model.decomposition:
                for p in self.model.model_trend.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model_res.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model_trend.TCN_backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model_res.TCN_backbone.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.model.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model.TCN_backbone.parameters():
                    p.requires_grad = False 
        if self.online == 'regressor_tcn':
            if self.model.decomposition:
                for p in self.model.model_trend.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model_res.backbone.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.model.backbone.parameters():
                    p.requires_grad = False 
        elif self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False
        elif self.online == 'encoder':
            if self.model.decomposition:
                for p in self.model.model_trend.head.parameters():
                    p.requires_grad = False 
                for p in self.model.model_res.head.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.model.head.parameters():
                    p.requires_grad = False 
        elif self.online == 'tcn':
            if self.model.decomposition:
                for p in self.model.model_trend.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model_res.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model_trend.head.parameters():
                    p.requires_grad = False 
                for p in self.model.model_res.head.parameters():
                    p.requires_grad = False 
            else:
                for p in self.model.model.backbone.parameters():
                    p.requires_grad = False 
                for p in self.model.model.head.parameters():
                    p.requires_grad = False 
        
        preds = []
        trues = []
        start = time.time()
        maes,mses,rmses,mapes,mspes = [],[],[],[],[]

        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader): batch_y is the predicted label
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape, mspe = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)
        MAE, MSE, RMSE, MAPE, MSPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, mspe: {}, mape: {}, time:{}'.format(mse, mae, mspe, mape, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode =='test' and self.online != 'none':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x)
        else:
            outputs = self.model(x)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return rearrange(outputs, 'b t d -> b (t d)'), rearrange(batch_y, 'b t d -> b (t d)')
    
    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        b, t, d = batch_y.shape
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()
        
        x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        for _ in range(self.n_inner):
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x)
            else:
                outputs = self.model(x)
            outputs = rearrange(outputs, 'b t d -> b (t d)').float().to(self.device)
            # loss = criterion(outputs[:, :d], true[:, :d])
            loss = criterion(outputs, true)
            # if not self.buffer.is_empty():
            #     buff_x, buff_y, idx = self.buffer.get_data(8)
            #     out = self.model(buff_x)
            #     out = rearrange(out, 'b t d -> b (t d)').float().to(self.device)
            #     loss += 0.5 * criterion(out, buff_y)
            # if self.aug > 0:
            #     loss_aug = 0
            #     for i in range(self.aug):
            #         batch_xa, _ = self.augmenter(x, torch.ones_like(batch_x).to(self.device))
            #         pred = self.model(batch_xa)
            #         pred = rearrange(pred, 'b t d -> b (t d)')
            #         loss_aug += criterion(pred, true)
            #     loss_aug /= self.aug
            #     loss += self.args.loss_aug * loss_aug
            loss.backward()
            self.opt.step()       
            self.model.store_grad()
            self.opt.zero_grad()

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        idx = self.count +  torch.arange(batch_y.size(0)).to(self.device)
        self.count += batch_y.size(0)
        self.buffer.add_data(examples = x, labels = true, logits = idx)
        return outputs, rearrange(batch_y, 'b t d -> b (t d)')

