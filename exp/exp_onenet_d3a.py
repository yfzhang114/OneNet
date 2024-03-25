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
from collections import OrderedDict, defaultdict
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils.buffer import Buffer

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split

import os
import time
from pathlib import Path
from exp.exp_patch import net as PatchTST
import warnings
warnings.filterwarnings('ignore')


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)

class net(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        encoder = TSEncoder(input_dims=args.seq_len,
                             output_dims=320,  # standard ts2vec backbone value
                             hidden_dims=64, # standard ts2vec backbone value
                             depth=10) 
        self.encoder_time = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.regressor_time = nn.Linear(320, args.pred_len).to(self.device)
        
        encoder = TSEncoder(input_dims=args.enc_in + 7,
                             output_dims=320,  # standard ts2vec backbone value
                             hidden_dims=64, # standard ts2vec backbone value
                             depth=10) 
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        
        self.dim = args.c_out * args.pred_len
        
        self.regressor = nn.Linear(320, self.dim).to(self.device)

    def forward_individual(self, x, x_mark):
        rep = self.encoder_time.encoder.forward_time(x)
        y = self.regressor_time(rep).transpose(1, 2)
        y0 = rearrange(y, 'b t d -> b (t d)')
        
        
        x = torch.cat([x, x_mark], dim=-1)
        rep2 = self.encoder(x)[:, -1]
        y2 = self.regressor(rep2)
    
        return y0, y2
    
    def forward_weight(self, x, x_mark, g0, g2):
        rep = self.encoder_time.encoder.forward_time(x)
        y = self.regressor_time(rep).transpose(1, 2)
        y0 = rearrange(y, 'b t d -> b (t d)')
        
        x = torch.cat([x, x_mark], dim=-1)
        rep2 = self.encoder(x)[:, -1]
        y2 = self.regressor(rep2)
    
        return y0 * g0 + y2 * g2, y0, y2
        
    def store_grad(self):
        for name, layer in self.encoder.named_modules():    
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
        for name, layer in self.encoder_time.named_modules():    
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
                loss = criterion(pred[0], true)+ criterion(pred[1], true)
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
            weight = 1. / len(pred)
            pred = pred[0] * weight  + weight * pred[1]
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
        elif self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False

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
        print('mse:{}, mae:{}, time:{}'.format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode =='test' and self.online != 'none':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)
        if mode =='test' and self.online == 'none':
            return self._ol_one_batch_(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = batch_x.float().to(self.device) #torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float()
        y0, y1 = self.model.forward_individual(x, batch_x_mark)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return [y0, y1], rearrange(batch_y, 'b t d -> b (t d)')
    
    def sleep_stage(self,):
        criterion = self._select_criterion()
        batch_size = min(self.args.batch_size, self.args.sleep_interval)
        steps = self.sleep_interval // batch_size
        for e in range(self.args.sleep_epochs):
            losses, losses_cons = [], []
            for _ in range(steps):
                buff_x, buff_y, logits = self.buffer.get_data(batch_size)
                outputs, y1, y2 = self.model.forward_weight(buff_x[:,:,:self.args.enc_in], buff_x[:,:,self.args.enc_in:], 0.5, 0.5)
                buff_y_target = rearrange(buff_y, 'b t d -> b (t d)').float()
                loss = criterion(y1, buff_y_target) + criterion(y2, buff_y_target)
                losses.append(loss.item())
                
                if self.args.offline_adjust != 0 and not self.buffer_adjust.is_empty():
                    buff_x_prev, buff_y_prev, logits_prev = self.buffer_adjust.get_data(batch_size)
                    x_edit, y_edit = self.get_adjust_data(buff_x, buff_y, buff_x_prev, buff_y_prev)
                    y_edit = rearrange(y_edit, 'b t d -> b (t d)').float()
                    logits, y1, y2 = self.model.forward_weight(x_edit[:,:,:self.args.enc_in], x_edit[:,:,self.args.enc_in:], 0.5, 0.5)
                    loss_adjust = self.args.online_adjust_var * criterion(y1, y_edit) + criterion(y2, y_edit)
                    del logits, x_edit, buff_x_prev, buff_y_prev, y_edit, logits_prev, outputs
                else:
                    loss_adjust = torch.tensor(0)
                    
                losses_cons.append(loss_adjust.item())
                loss += self.args.offline_adjust * loss_adjust
                
                # out = rearrange(out, 'b t d -> b (t d)')
                loss.backward()
                self.opt.step() 
                self.model.store_grad()
                self.opt.zero_grad()

            print(f'Sleep stage: epoch {e} loss {np.mean(losses)} loss_consistence {np.mean(losses_cons)}')

        self.buffer_adjust = self.buffer
        self.buffer = Buffer(self.sleep_interval, self.device)  
        torch.cuda.empty_cache()
    
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
        return x_edit, y_edit
    
    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()
        
        x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if self.args.online_adjust != 0 and not self.buffer.is_empty():
            buff_x, buff_y, logits = self.buffer.get_data(self.args.batch_size)
            x_edit, y_edit = self.get_adjust_data(torch.cat([x, batch_x_mark], dim=-1), batch_y, buff_x, buff_y)
            y_edit = rearrange(y_edit, 'b t d -> b (t d)').float()
            # y_edit = self.get_adjust_data(torch.cat([true, buff_y], dim=0), is_label=True)
            logits, y1, y2 = self.model.forward_weight(x_edit[:,:,:self.args.enc_in], x_edit[:,:,self.args.enc_in:], 0.5, 0.5)
            loss_adjust = self.args.online_adjust_var * criterion(y1, y_edit) + criterion(y2, y_edit)
            del logits, x_edit, buff_x, buff_y, y1, y2
        else:
            loss_adjust = 0

        outputs, y1, y2 = self.model.forward_weight(x, batch_x_mark, 1./2, 1./2)

        l1, l2 = criterion(y1, true), criterion(y2, true)
        loss = l1 + l2 + self.args.online_adjust * loss_adjust
        
        if self.online != 'none':
            loss.backward()
            self.opt.step()    
            self.model.store_grad()
            self.opt.zero_grad()

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        if self.sleep_interval > 1  or self.args.online_adjust > 0:
            self.detector.add_data(loss.item(), batch_x)
            self.count += batch_y.size(0)
            self.buffer.add_data(examples = torch.cat([x, batch_x_mark], dim=-1), labels = batch_y, logits = outputs.data)
            status, name = self.detector.run_test()
            if (status == 1 or self.detector.cnt >= 1000) and self.sleep_interval > 1:
                self.sleep_stage()
                self.detector.reset()
        torch.cuda.empty_cache()
        return outputs, rearrange(batch_y, 'b t d -> b (t d)')