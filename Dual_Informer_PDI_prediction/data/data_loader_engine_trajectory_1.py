# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.utils import shuffle
from torch.utils.data import Dataset

class HILabeling(object):
    
    def __init__(self,
                  root_path,
                  dataset_name='FD001',
                  HI_labeling_style='HI_pw_quadratic',
                  **kwargs):
        
        self.root_path = root_path
        self.data = pd.DataFrame([])
        self.dataset_name = dataset_name
        self.HI_labeling_style = HI_labeling_style
        self.__process__()
    
    # data load
    def loader_engine(self, **kwargs):
        path = os.path.join(self.root_path,'{}/train_{}.csv'.format(self.dataset_name,self.dataset_name))
        self.data = pd.read_csv( path ,header=None, **kwargs)
 
    def add_columns_name(self):
        sensor_columns = ["sensor {}".format(s) for s in range(1,22)]
        info_columns = ['unit_id','cycle']
        settings_columns = ['setting 1', 'setting 2', 'setting 3']
        self.data.columns = info_columns + settings_columns + sensor_columns
           
        
    def labeling(self,piecewise_point):
        ##for train
        maxRUL_dict = self.data.groupby('unit_id')['cycle'].max().to_dict()
        self.data['maxRUL'] = self.data['unit_id'].map(maxRUL_dict)
        self.data['RUL'] = self.data['maxRUL'] - self.data['cycle']
        
        #linear
        self.data['HI_linear'] = 1 - self.data['cycle']/self.data['maxRUL']
        #piece_wise linear
        FPT = self.data['maxRUL'] - piecewise_point
        self.data['HI_pw_linear'] =  self.data['cycle']/ (FPT - self.data['maxRUL'] ) + self.data['maxRUL']/(self.data['maxRUL'] - FPT)
        filter_piece_wise = (self.data['cycle'] <= FPT)
        self.data.loc[filter_piece_wise,['HI_pw_linear']] = 1
        #quadratic
        self.data['HI_quadratic'] = 1 -((self.data['cycle']*self.data['cycle'])/(self.data['maxRUL']*self.data['maxRUL']))
        #piece_wise quadratic
        self.data['HI_pw_quadratic'] = 1 - ((1/(piecewise_point**2))*(self.data['cycle']-FPT)**2)
        filter_piece_wise = (self.data['cycle'] <= FPT)
        self.data.loc[filter_piece_wise,['HI_pw_quadratic']] = 1
        
        self.data['HI'] = self.data[self.HI_labeling_style]
        #self.data.drop(['maxRUL','HI_linear','HI_pw_linear','HI_quadratic','HI_pw_quadratic'],axis=1,inplace=True)
        self.data.drop(['HI_linear','HI_pw_linear','HI_quadratic','HI_pw_quadratic'],axis=1,inplace=True)

    
    
    def __process__(self):
        self.loader_engine()
        self.add_columns_name()
        self.labeling(piecewise_point=125)
        self.data.to_csv(self.root_path + '/{}/train_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=False)
        


class DataReaderTrajactory(Dataset):
    
    def __init__(self, root_path,sensor_features,normal_style, flag='pred', size=None, 
                 features='MS', dataset_name='FD001', HI_labeling_style='HI_pw_quadratic',
                 target='HI', scale=True, inverse=False, timeenc=0, freq='15min',           
                 cols=None):

        self.HI_labeling_style = HI_labeling_style
        self.dataset_name = dataset_name
        self.sensor_features = sensor_features
        self.normal_style = normal_style
        
    
        # info
        self.root_path = root_path
        self.data_path = '{}/train_{}.csv'.format(self.dataset_name,self.HI_labeling_style)

        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.inverse = inverse
        
        self.data_x = pd.DataFrame(data=[])
        self.data_y = pd.DataFrame(data=[])
        self.all_seq_x = []
        self._all_seq_y = []

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        

        self.__read_data__(is_padding=True)  # data preprocessing
              
    
    def __read_data__(self,is_padding):
          
        '''
        1. spilt for data,val,test according to trajectory
        2. normalization
        3. prepare the data for encoder and decoder according to trajectory   (self.transform_data)
        '''
        
        # load data    
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path),header=0,index_col=["unit_id"])
        

        df_raw = df_raw.loc[df_raw['maxRUL']>=187]
        test = df_raw.loc[(df_raw['RUL']<=187)&(df_raw['RUL']>=40)]
        df_raw = df_raw.loc[(df_raw['RUL']<=187)&(df_raw['RUL']>=40)]
        
        train_turbines = np.arange(len(df_raw.index.to_series().unique()))     
          
        train_turbines, vali_turbines = train_test_split(train_turbines, test_size=0.3,random_state =1334)
       
         
        idx_train = df_raw.index.to_series().unique()[train_turbines]
        idx_validation = df_raw.index.to_series().unique()[vali_turbines]
        train = df_raw.loc[idx_train]
        validation = df_raw.loc[idx_validation]

              
        # normalization
        if self.normal_style == 'StandardScaler':
            self.scaler = StandardScaler()
        elif self.normal_style == 'MinMaxScale':
            self.scaler = MinMaxScaler()
            
        train_normalize = train.copy()
        validation_normalize = validation.copy()
        test_normalize = test.copy()
        
        self.scaler.fit(train.loc[:,self.sensor_features].values)     # use the mean and variance in data dataset, do not use the mean and variance of the whole data
        train_normalize.loc[:,self.sensor_features] = self.scaler.transform(train.loc[:,self.sensor_features].values)  #cycle和HI 不标准化
        validation_normalize.loc[:,self.sensor_features]= self.scaler.transform(validation.loc[:,self.sensor_features].values)
        test_normalize.loc[:,self.sensor_features] = self.scaler.transform(test.loc[:,self.sensor_features].values)
        
        train =  train_normalize.copy()
        validation = validation_normalize.copy()
        test = test_normalize.copy()
        del (train_normalize,validation_normalize,test_normalize) 

        useful_columns = ['cycle'] + self.sensor_features + ['HI']
        train = train.loc[:,useful_columns]
        validation = validation.loc[:,useful_columns]        
        test = test.loc[:,useful_columns]


        train.to_csv(self.root_path + '/{}/train_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        validation.to_csv(self.root_path+ '/{}/validation_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        test.to_csv(self.root_path + '/{}/test_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        

        if self.flag =='train':
            self.data_x = train
        elif self.flag =='val':
            self.data_x = validation
        elif self.flag =='test':
            self.data_x = test
                
        if self.inverse:
            self.data_y = self.data_x
        else:
            self.data_y = self.data_x
            
                    
        ## prepare the data for encoder and decoder according to trajectory:  96*14； 72*14
        self.all_seq_x, self.all_seq_y = self.transform_data()


        
    def transform_data(self):
        ### enc, dec for save the precessed data(time window) 96*14 
        enc,dec = [],[]
                      
        #Loop through each trajectory
        num_units = len(self.data_x.index.to_series().unique())
        print('{} trajectories in {} dataset'.format(num_units,self.flag))
        
        num_units = 0     
        for index in (self.data_x.index.to_series().unique()): 
            #get the whole trajectory (index)
            temp_df = pd.DataFrame(self.data_x.loc[index]) 
            
            # Loop through the data in the object (index) trajectory
            data_enc_npc, data_dec_npc, array_data_enc, array_data_dec = [],[],[],[]
            len_trajectory = len(temp_df)
            
            # enc_last_index = len_trajectory - self.pred_len
            enc_last_index = len_trajectory
            

            for i in range(enc_last_index - self.seq_len + 1):
                s_begin = i
                s_end = s_begin + self.seq_len
                # r_begin = s_end - self.label_len 
                # r_end = r_begin + self.label_len + self.pred_len 
  
                data_enc_npc = temp_df.iloc[s_begin:s_end]
                # data_dec_npc = temp_df.iloc[r_begin:r_end]
       
                array_data_enc.append(data_enc_npc)
                #array_data_dec.append(data_dec_npc)
            
            enc = enc + array_data_enc
            #dec = dec + array_data_dec
                        
        #return enc,dec
        return enc,enc
    
                

    def __getitem__(self,index):  

        # seq_x = self.all_seq_x[index]
        # seq_y = self.all_seq_y[index]
        
        seq_x = self.all_seq_x[index].drop(['cycle'],axis=1).values
        seq_y = self.all_seq_y[index].drop(['cycle'],axis=1).values
        
            
        return seq_x, seq_y
        
    def __len__(self):
        '''   
        Returns: len(self.all_seq_x) 
        TYPE: int
              #seq_len * enc_in （96*14）的数目，有多少个训练数据(时间窗)
        '''
        
        return len(self.all_seq_x)   
    
    def inverse_transform(self, data):
        
        return self.scaler.inverse_transform(data)
    


