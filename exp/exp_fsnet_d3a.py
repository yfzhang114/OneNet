from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.fsnet import TSEncoder, GlobalLocalMultiscaleTSEncoder
from models.ts2vec.losses import hierarchical_contrastive_loss
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cumavg
from scipy.stats import norm
import numpy as np
from einops import rearrange
from collections import OrderedDict, defaultdict
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from utils.buffer import Buffer

import os
import time
from pathlib import Path
from copy import deepcopy


import warnings
warnings.filterwarnings('ignore')


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)[:, -1]

class net(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        encoder = TSEncoder(input_dims=args.enc_in + 7,
                             output_dims=320,  # standard ts2vec backbone value
                             hidden_dims=64, # standard ts2vec backbone value
                             depth=10) 
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.dim = args.c_out * args.pred_len
        
        #self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dim)).to(self.device)
        self.regressor = nn.Linear(320, self.dim).to(self.device)
        
    def forward(self, x, return_feature=False):
        rep = self.encoder(x)
        y = self.regressor(rep)
        if return_feature:
            return y, rep
        return y
    def store_grad(self):
        for name, layer in self.encoder.named_modules():    
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
        
class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = net(args, device = self.device)
         
        self.sleep_interval = args.sleep_interval
        self.sleep_kl_pre = args.sleep_kl_pre
        
        buff_size = args.sleep_interval if args.sleep_interval > 0 else 100
        self.buffer = Buffer(buff_size, self.device, mode='fifo')  # FIFO
        self.count, self.buffer_adjust, self.ema_model = 0, Buffer(256, self.device, mode='fifo'), None
        from utils.detector import STEPD
        self.detector = STEPD(new_window_size=buff_size, alpha_w=args.alpha_w, alpha_d=args.alpha_d)
            
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
            delay_fb=args.delay_fb,
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
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                self.opt.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = 0.

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
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
        self.model.eval()
        
        if self.online == 'regressor':
            for p in self.model.encoder.parameters():
                p.requires_grad = False 
        # elif self.online == 'none':
        #     for p in self.model.parameters():
        #         p.requires_grad = False
        
        preds = []
        trues = []
        start = time.time()
        maes,mses,rmses,mapes,mspes = [],[],[],[],[]
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
        print('mse:{}, mae:{}, time:{}, sleep times {}'.format(mse, mae, exp_time, self.detector.shift_cnt))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode =='test':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)
        batch_y = batch_y.float()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x)
        else:
            outputs = self.model(x)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
    
        return outputs, rearrange(batch_y, 'b t d -> b (t d)')
    
    def update_ema_variables(self, ema_model, model, alpha_teacher=0.99): #, iteration):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model

    def sleep_stage(self,):
        criterion = self._select_criterion()
        
        batch_size = min(self.args.batch_size, self.args.sleep_interval)
        steps = self.sleep_interval // batch_size
        for e in range(self.args.sleep_epochs):
            losses, losses_cons = [], []
            for _ in range(steps):
                buff_x, buff_y, logits = self.buffer.get_data(batch_size)
                
                out = self.model(buff_x.detach())
                
                if self.args.offline_adjust != 0 and not self.buffer_adjust.is_empty():
                    buff_x_prev, buff_y_prev, logits_prev = self.buffer_adjust.get_data(batch_size)
                    x_edit, y_edit = self.get_adjust_data(buff_x, buff_y, buff_x_prev, buff_y_prev)
                    y_edit = rearrange(y_edit, 'b t d -> b (t d)').float()
                    # y_edit = self.get_adjust_data(torch.cat([true, buff_y], dim=0), is_label=True)
                    logits = self.model(x_edit.detach())
                    loss_adjust = criterion(logits, y_edit)
                    del logits, x_edit, buff_x_prev, buff_y_prev, y_edit
                else:
                    loss_adjust = torch.tensor(0)
                    
                losses_cons.append(loss_adjust.item())
                
                buff_y = rearrange(buff_y, 'b t d -> b (t d)').float()
                loss = criterion(out, buff_y) 
                losses.append(loss.item())
                
                loss += self.args.offline_adjust * loss_adjust
                
                # out = rearrange(out, 'b t d -> b (t d)')
                loss.backward()
                self.opt.step() 
                self.model.store_grad()
                self.opt.zero_grad()

            print(f'Sleep stage: epoch {e} loss {np.mean(losses)} loss_consistence {np.mean(losses_cons)}')

        self.buffer_adjust = self.buffer
        self.buffer = Buffer(self.sleep_interval, self.device)  
    
    def get_adjust_data(self, x, y, buff_x, buff_y):
        n1, n2 = x.shape[0], buff_x.shape[0]
        l, h = x.shape[1], y.shape[1]
        x_data, x_date = x[:,:,:self.args.enc_in], x[:,:,self.args.enc_in:]
        buff_x_data, buff_x_date = buff_x[:,:,:self.args.enc_in], buff_x[:,:,self.args.enc_in:]

        x_data = torch.cat([x_data, y], dim=1)
        buff_x_data = torch.cat([buff_x_data, buff_y], dim=1)
                
        mean_x, mean_buff_x = torch.mean(x_data, dim=1, keepdim=True).detach(), torch.mean(buff_x_data, dim=1, keepdim=True).detach()
        stdev_x, stdev_buff_x = torch.sqrt(torch.var(x_data, dim=1, keepdim=True, unbiased=False) + 1e-5).detach(), torch.sqrt(torch.var(buff_x_data, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        
        if n1 > 1:
            mean_x, stdev_x = torch.mean(mean_x, dim=0, keepdim=True), torch.mean(stdev_x, dim=0, keepdim=True)
        mean_x = mean_x.repeat(n2,1,1)
        stdev_x = stdev_x.repeat(n2,1,1)
        
        U = torch.normal(0, stdev_x)
        buff_x_data = buff_x_data + self.args.var_weight * U
        
        x_edit = torch.cat([buff_x_data[:,:l,:], buff_x_date], dim=-1)
        y_edit = buff_x_data[:,l:,:]
        return x_edit, buff_y

    def is_outlier(self, value):
        # 使用标准差的方法检测异常值
        mean_value = torch.mean(value, dim=1, keepdim=True)
        if value.shape[1] == 1:
            return torch.ones_like(value).bool()
        std_dev_value = torch.std(value, dim=1, keepdims=True)
        z_score = (value - mean_value) / (std_dev_value + 1e-4)
        warning_threshold = norm.ppf(1 - self.args.alpha_d / 2)
        return torch.abs(z_score) < warning_threshold
    
    def plt_mask(self, x, mask):
        import matplotlib.pyplot as plt
        x, mask = x[0].cpu(), mask[0].cpu()
        # 绘制折线图
        plt.plot(x, label='x')

        # 标注 mask=True 的位置
        mask_indices = torch.nonzero(mask).squeeze().numpy()
        plt.scatter(mask_indices, x[mask_indices], color='red', label='mask=True')

        # 添加标签和图例
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Line Plot with Mask')
        plt.legend()
        plt.savefig('mask.pdf')
        plt.close()
    
    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()
        
        x = torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)
        batch_y = batch_y.float().to(self.device)
        for _ in range(self.n_inner):
            if self.online == 'none':
                with torch.no_grad():
                    outputs = self.model(x)
            else:
                if self.args.online_adjust != 0 and not self.buffer.is_empty():
                    buff_x, buff_y, logits = self.buffer.get_data(self.args.batch_size)
                    x_edit, y_edit = self.get_adjust_data(x, batch_y, buff_x, buff_y)
                    y_edit = rearrange(y_edit, 'b t d -> b (t d)').float()
                    logits = self.model(x_edit.detach())
                    loss_adjust = criterion(logits, y_edit)
                    del logits, x_edit, buff_x, buff_y
                else:
                    loss_adjust = 0

                outputs = self.model(x)
                loss = criterion(outputs, true) + self.args.online_adjust * loss_adjust
                loss.backward()
                self.opt.step()       
                self.model.store_grad()
                self.opt.zero_grad()

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        
        if self.sleep_interval > 1  or self.args.online_adjust > 0:
            self.detector.add_data(loss.item(), batch_x)
            self.count += batch_y.size(0)
            self.buffer.add_data(examples = x, labels = batch_y, logits = outputs.data)
            # self.buffer_adjust.add_data(examples = x, labels = batch_y, logits = outputs.data)
            status, name = self.detector.run_test()
            if (status == 1 or self.detector.cnt >= 1000) and self.sleep_interval > 1:
                self.sleep_stage()
                self.detector.reset()
        torch.cuda.empty_cache()
        return outputs, rearrange(batch_y, 'b t d -> b (t d)')
