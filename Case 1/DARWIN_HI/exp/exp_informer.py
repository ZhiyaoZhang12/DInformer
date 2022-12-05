# -*- coding: utf-8 -*-
####DARWIN HI
import os
import torch
import numpy as np
import pandas as pd
from models.model import Informer, InformerStack,BiLSTM,DCNN,DH_1,TransGCU#,DeepHealth
from models.DAST_Network import DAST
from exp.exp_basic import Exp_Basic
from data_loader import DataReaderTrajactory,HILabeling
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)       
        self._HI_()  
    

    def _HI_(self):
        args = self.args
        hilabeling = HILabeling(root_path=args.root_path,
                   data_path=args.data_path,           
                   HI_labeling_style = args.HI_labeling_style,
                   dataset_name=args.dataset_name,
                   pred_len=args.pred_len,seq_len=args.seq_len)
        self.train_data, self.test_data = hilabeling.process()
        
        
    def _build_model(self):

        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
            'biLSTM':BiLSTM,
            'dcnn':DCNN,
            'transgcu':TransGCU,
            'dh_1':DH_1,
            'DAST':DAST,
            #'deephealth':DeepHealth,
        }
        
        if self.args.model_name=='informer' or self.args.model_name=='informerstack':
            e_layers = self.args.e_layers if self.args.model_name=='informer' else self.args.s_layers
            model = model_dict[self.args.model_name](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.is_perception,
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
                
        elif self.args.model_name == 'biLSTM':
            model = model_dict[self.args.model_name](input_size=self.args.enc_in,hidden_size=512,
                                                num_layers=5,output_size=self.args.c_out,seq_len=self.args.seq_len,
                                                out_len=self.args.pred_len).float() #hidden_size=512,num_layers=5
            
        elif self.args.model_name == 'dcnn':
            model = model_dict[self.args.model_name](pred_len=self.args.pred_len).float()
        
        elif self.args.model_name =='transgcu':
            e_layers = self.args.e_layers if self.args.model_name=='transgcu' else self.args.s_layers
            model = model_dict[self.args.model_name](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.is_perception,
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
            
            
        elif self.args.model_name =='dh_1':
            e_layers = self.args.e_layers if self.args.model_name=='dh_1' else self.args.s_layers
            model = model_dict[self.args.model_name](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.is_perception,
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
            
        
        elif self.args.model_name == 'DAST':
            d_model = 512
            model = model_dict[self.args.model_name](dim_val_s=self.args.d_model,
                                                dim_attn_s=self.args.d_model, 
                                                dim_val_t=self.args.d_model,
                                                dim_attn_t=self.args.d_model,
                                                dim_val=self.args.d_model, 
                                                dim_attn=self.args.d_model, 
                                                time_step=self.args.seq_len,
                                                input_size=self.args.enc_in, 
                                                dec_seq_len=self.args.label_len, 
                                                out_seq_len=self.args.pred_len, 
                                                n_decoder_layers = 1, 
                                                n_encoder_layers = 2,
                                                n_heads = 8, #8,4
                                                dropout = 0.1
                                                ).float()            
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model
    
        

    def _get_data(self, flag):
        args = self.args
      
        Data =  DataReaderTrajactory 
             
        if flag == 'test_window':       
            shuffle_flag = False; drop_last = False; batch_size = 25; freq=args.detail_freq    #drop_last = False;batch_size = 25
        elif flag=='test_whole':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.detail_freq     
        elif flag=='pred':   
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
        else:  
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
       
        
        data_set = Data(
            root_path=args.root_path,
            rate_data=args.rate_data,
            train_data=self.train_data,
            test_data=self.test_data,
            sensor_features=args.sensor_features,
            is_padding=args.is_padding,
            is_descrsing=args.is_descrsing,
            data_augmentation=args.data_augmentation,
            HI_labeling_style=args.HI_labeling_style,
            normal_style = args.normal_style,
            synthetic_data_path = args.synthetic_data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            freq=freq,
            cols=args.cols
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
            
            if self.args.output_attention:
                pred, true,attn_weights = self._process_one_batch(vali_data, batch_x, batch_y)
            else:
                pred, true = self._process_one_batch(vali_data, batch_x, batch_y)
                     
            #loss = criterion(pred.detach().cpu(), true.detach().cpu())    ###loss.detach().cpu(), else cuda out memory
            if self.args.is_perception == False:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
            elif self.args.is_perception == True:
                loss = criterion(pred[:,-self.args.seq_len:,:].detach().cpu(), true[:,-self.args.seq_len:,:].detach().cpu())
                            
            total_loss.append(loss)
            
        #AttributeError: 'torch.dtype' object has no attribute 'type'    
        #total_loss = np.average(total_loss)
        total_loss = torch.mean(torch.stack(total_loss))
        
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data_window, test_loader_window = self._get_data(flag = 'test_window')
        test_data_whole, test_loader_whole = self._get_data(flag = 'test_whole')
        
        
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
                
                if self.args.output_attention:
                    pred, true,attn_weights = self._process_one_batch(train_data, batch_x, batch_y)
                else:
                    
                    pred, true = self._process_one_batch(train_data, batch_x, batch_y)
                    #print('pred',pred.shape())
                    #print('true',true.shape())
                
                if self.args.is_perception == False:
                    loss = criterion(pred, true)
                elif self.args.is_perception == True:
                    loss = criterion(pred[:,-self.args.seq_len:,:], true[:,-self.args.seq_len:,:])
            
                
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
            test_loss_whole = self.vali(test_data_whole, test_loader_whole, criterion)
            test_loss_window = self.vali(test_data_window, test_loader_window, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss_whole: {4:.7f}  Test Loss_window: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss_whole,test_loss_window))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting,flag):
        test_data, test_loader = self._get_data(flag=flag)
      
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y) in enumerate(test_loader):
            
            if self.args.output_attention:
                pred, true,attn_weights = self._process_one_batch(test_data, batch_x, batch_y)
            else:
                pred, true = self._process_one_batch(test_data, batch_x, batch_y)
           
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            

        preds = np.array(preds)
        trues = np.array(trues)

        #print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = self.args.root_path + '/results/{}/'.format(self.args.dataset_name) + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        #
        if self.args.is_perception == True:
            preds1 = preds[:,self.args.label_len:,:]
            trues1 = trues[:,self.args.label_len:,:]
            mae1, mse1, rmse1, mape1, mspe1 = metric(preds1, trues1)
            
            preds2 = preds[:,-self.args.seq_len:,:]
            trues2 = trues[:,-self.args.seq_len:,:]
            mae2, mse2, rmse2, mape2, mspe2 = metric(preds2, trues2)
        
            print('perception mse:{}, mae:{}'.format(mse1, mae1))
            print('pred mse:{}, mae:{}'.format(mse2, mae2))
            
            #np.save(folder_path+'metrics1_{}.npy'.format(flag), np.array([mae1, mse1, rmse1, mape1, mspe1]))
            #np.save(folder_path+'pred1_{}'.format(flag), preds1)
            #np.save(folder_path+'true1_{}'.format(flag), trues1)
            
            #np.save(folder_path+'metrics2_{}.npy'.format(flag), np.array([mae2, mse2, rmse2, mape2, mspe2]))
            #np.save(folder_path+'pred2_{}'.format(flag), preds2)
            #np.save(folder_path+'true2_{}'.format(flag), trues2)
            
            #save results as dataframe
            if flag == 'test_window':
                columns_name = ['HI']
                pred_df = pd.DataFrame(preds.reshape(preds.shape[0]*preds.shape[1],preds.shape[2]),columns=columns_name)
                true_df = pd.DataFrame(trues.reshape(trues.shape[0]*trues.shape[1],trues.shape[2]),columns=columns_name)
        
                unit_df = np.array([[i+1]*(self.args.label_len + self.args.pred_len) for i in range(100)]).reshape(-1,1) 
                pred_df['Unit_id'] = unit_df
                true_df['Unit_id'] = unit_df
                true_df.to_csv(folder_path+'/true_data.csv',header=True,index=False)
                pred_df.to_csv(folder_path+'/pred_data.csv',header=True,index=False) 
              
        
        # mae, mse, rmse, mape, mspe = metric(preds, preds)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics_{}.npy'.format(flag), np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred_{}.npy'.format(flag), preds)
        np.save(folder_path+'true_{}.npy'.format(flag), trues)
        
        if self.args.output_attention:
            np.save(folder_path+'attn_weights.npy', attn_weights)   
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
        if self.args.model_name in ['informer' ,'informerstack' ,'transgcu']:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x = batch_x.float()[:,:,:-1]    #delect HI

            # decoder input
            if self.args.padding==0:
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            elif self.args.padding==1:
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            dec_inp = dec_inp.float()[:,:,:-1]   #delect HI

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x,  dec_inp)[0]
                        attn_weights = self.model(batch_x,  dec_inp)[1]
                    else:
                        outputs = self.model(batch_x,  dec_inp)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x,  dec_inp)[0]
                    attn_weights = self.model(batch_x,  dec_inp)[1]
                else:
                    outputs = self.model(batch_x,  dec_inp)
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)


            #batch_y is correspond groud_truth
            if self.args.features =='MS' or self.args.features =='S':
                if self.args.is_perception == False:
                    batch_y = batch_y[:,-self.args.pred_len:,-1:].to(self.device)     #the last column is HI
                elif self.args.is_perception == True:
                    batch_y = batch_y[:,(-self.args.pred_len-self.args.label_len):,-1:].to(self.device) 

            elif self.args.features =='M':
                if self.args.is_perception == False:
                    batch_y = batch_y[:,-self.args.pred_len:,:-1].to(self.device)     #remain sensor, exclude HI
                elif self.args.is_perception == True:
                    batch_y = batch_y
                    
            if self.args.output_attention:
                return outputs, batch_y,attn_weights
            else:
                return outputs, batch_y

        elif self.args.model_name == 'biLSTM':
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()[:,-self.args.pred_len:,-1:].to(self.device)

            batch_x = batch_x.float()[:,:,:-1]   #delect HI
            outputs = self.model(batch_x)
            
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)

            return outputs, batch_y
        
        
        elif self.args.model_name == 'dcnn':
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()[:,-self.args.pred_len:,-1:].to(self.device)

            batch_x = batch_x.float()[:,:,:-1]   #delect HI
            batch_x = batch_x.unsqueeze(1) #32*48*14 --> 32*1*48*14            
            outputs = self.model(batch_x)
                
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)
            
            return outputs, batch_y
        
        
        elif self.args.model_name == 'DAST':  #11.30
            batch_x = batch_x.float().to(self.device)[:,:,:-1]   #delect HI
            batch_y = batch_y.float().to(self.device)[:,-self.args.pred_len:,-1:].to(self.device)          

            outputs = self.model(batch_x)
            outputs = outputs.reshape(outputs.shape[0],outputs.shape[1],1) #增加维度 hi要占位  (bs,pl) --> (bs,pl,1)
            return outputs, batch_y 
