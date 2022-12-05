# -*- coding: utf-8 -*-
####DARWIN AG DS02
import os
import torch
import numpy as np
import pandas as pd
from models.model import Informer, InformerStack
from exp.exp_basic import Exp_Basic
from datapreprocess import DataPreproserse
from dataloader import DataReader
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import warnings
import h5py
warnings.filterwarnings('ignore')

'''
test()保存构造的ag trajectory，并用真实hi替换，h5py格式
'''

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
        self.args = args
        
        self._datapreproserse_()  
    

    def _datapreproserse_(self):
        args = self.args
        datapreproserse = DataPreproserse(                       
                    root_path=args.root_path,
                    data_path=args.data_path,
                    dataset_name=args.dataset_name,
                    validation_split=args.validation_split,
                    normal_style=args.normal_style,
                    down_sampling=args.down_sampling,
                    down_sampling_rate=args.down_sampling_rate,
                    stride=args.stride,
                    seq_len=args.seq_len,
                    label_len=args.label_len,
                    pred_len=args.pred_len,
                     )
        
        self.train_enc, self.train_dec, self.val_enc, self.val_dec, self.test_enc, self.test_dec = datapreproserse.process()
        
        
    def _build_model(self):

        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
                
            
                     
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model
    
        

    def _get_data(self, flag):
        args = self.args
             
        if flag == 'test':       
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.detail_freq   
        elif flag=='pred':   
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        elif flag in ['train','val']: 
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
       
    
        data_set = DataReader(         
                    train_enc=self.train_enc,
                    train_dec=self.train_dec,
                    val_enc=self.val_enc,
                    val_dec=self.val_dec,
                    test_enc=self.test_enc,
                    test_dec=self.test_dec,
                    flag=flag,   
                    )
        
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader
    

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion =  nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion =  nn.L1Loss()
        return criterion
    

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []

        for i, (batch_x,batch_y,) in enumerate(vali_loader):
            synthetic, true = self._process_one_batch(
                vali_data, batch_x, batch_y)
            # loss = criterion(synthetic.detach().cpu(), true.detach().cpu())
            
            #2.18不算最后一列HI的loss
            loss = criterion(synthetic[:,:,:-1].detach().cpu(),true[:,:,:-1].detach().cpu())
                                          
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                synthetic, true = self._process_one_batch(
                    train_data, batch_x, batch_y)
                

                # loss = criterion(synthetic, true)
                
                #2.18不算最后一列HI的loss
                loss = criterion(synthetic[:,:,:-1],true[:,:,:-1])
                
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        synthetics = []
        trues = []
        
        for i, (batch_x,batch_y) in enumerate(test_loader):
            synthetic, true = self._process_one_batch(test_data, batch_x, batch_y)
            synthetics.append(synthetic.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        synthetics = np.array(synthetics)
        trues = np.array(trues)
        synthetics = synthetics.reshape(-1, synthetics.shape[-2], synthetics.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        #mae, mse, rmse, mape, mspe = metric(synthetics, trues)
        mae, mse, rmse, mape, mspe = metric(synthetics[:,:,:-1], trues[:,:,:-1])  #不计算HI那一列的误差
        print('mse:{}, mae:{}'.format(mse, mae))
        
        
        ##np.save(self.args.root_path +  '/results/{}_metrics.npy'.format(setting),np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path+'synthetic.npy', synthetics)
        #np.save(folder_path+'true.npy', trues)
             
        ##save ag data
        #trajectory的长度：synthetics.shape[1] 
        #trajectory的条数： synthetics.shape[0]
        #d_features+1(hi) = #synthetics.shape[2]
        synthetic_df = pd.DataFrame(synthetics.reshape(synthetics.shape[0]*synthetics.shape[1],synthetics.shape[2])) 
        trues_df = pd.DataFrame(trues.reshape(trues.shape[0]*trues.shape[1],trues.shape[2])) 
        unit_list = []
        for i in range(synthetics.shape[0]):
            unit_list = unit_list.append(np.ones(synthetics.shape[1])*(1000+i)) #不要跟原数据的unit_id数据重复了
        
        synthetic_df.insert(0,'unit',unit_list,allow_duplicates=False)
        # Create a new file hdf5 dataset
        f1 = h5py.File(self.args.root_path + '/results/'+'Raw_Synthetics_DS02_{}_rate{}.h5'.format(self.down_sampling,self.down_sampling_rate), 'w')
        f1.create_dataset('raw_ag_trajectories', data=np.arange(synthetic_df))
        f1.close()
       
        #用true的HI填充synthetic的HI 
        synthetic_df.drop(synthetic_df.iloc[:,-1],inplace=True,axis=1)
        synthetic_df.insert(-1,'HI',trues_df.iloc[:,-1],allow_duplicates=False)
        
        # Create a new file hdf5 dataset
        f = h5py.File(self.args.root_path + '/results/'+'Synthetics_DS02_{}_rate{}.h5'.format(self.down_sampling,self.down_sampling_rate), 'w')
        f.create_dataset('ag_trajectories', data=np.arange(synthetic_df))
        f.close()
        return
    
    

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y) in enumerate(pred_loader):
            
            if self.args.output_attention:
                pred, true,attn_weights = self._process_one_batch(pred_data, batch_x, batch_y)
            else:
                pred, true = self._process_one_batch(pred_data, batch_x, batch_y)
            
            
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = 'hi/results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    
    def _process_one_batch(self, dataset_object, batch_x, batch_y):
        if self.args.model=='informer' or self.args.model=='informerstack' or 'transgcu':
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()  #后面还要处理 


            # decoder input
            if self.args.padding==0:
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            #dec_inp = dec_inp.float()[:,:,:-1]   #delect HI   ag不需要删除hi

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x,  dec_inp)[0]
                        attn_weights = self.model(batch_x,  dec_inp)
                    else:
                        outputs = self.model(batch_x,  dec_inp)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x,  dec_inp)[0]
                    attn_weights = self.model(batch_x,  dec_inp)
                else:
                    outputs = self.model(batch_x,  dec_inp)
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)

            
            f_dim = -1 if self.args.features=='MS' else 0        ###f_dim=-1只取最后一列的数据，也就是HI

            # batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

            ###数据增强  输出的长度等于seq_len，batch_y是真实数据true
            batch_y = batch_y[:,-self.args.seq_len:,f_dim:].to(self.device)    #HI那一列也要  ag时,f_dim=0 就是全都要
            ###这里的output就是预测的synthetic

            return outputs, batch_y