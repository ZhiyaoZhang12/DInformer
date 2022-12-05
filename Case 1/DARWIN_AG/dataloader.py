# -*- coding: utf-8 -*-
###DARWIN  AG
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
                  data_path,
                  sensor_features,
                  normal_style,
                  dataset_name='FD001',
                  HI_labeling_style='HI_pw_quadratic',
                  **kwargs):
        
        self.root_path = root_path
        self.data_path = data_path
        self.sensor_features = sensor_features
        self.normal_style = normal_style
        self.data = pd.DataFrame([])
        self.dataset_name = dataset_name
        self.HI_labeling_style = HI_labeling_style
    
    # data load
    def loader_engine(self, **kwargs):
        path = os.path.join(self.data_path,'{}/train_{}.csv'.format(self.dataset_name,self.dataset_name))
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
        
                           
    def part_data(self):
        ###数据增强，只保留部分数据，RUL70-197 (HI=0.8-1)
        self.data = self.data.loc[self.data['maxRUL']>=187]
        self.data = self.data.loc[(self.data['RUL']<=187)&(self.data['RUL']>=40)]
        
        train_turbines = np.arange(len(self.data.index.to_series().unique()))     
        train_turbines, vali_turbines = train_test_split(train_turbines, test_size=0.3,random_state =1334)
       
        idx_train = self.data.index.to_series().unique()[train_turbines]
        idx_validation = self.data.index.to_series().unique()[vali_turbines]
        train = self.data.loc[idx_train]
        validation = self.data.loc[idx_validation]
        test = self.data
        return train, validation, test
    
    
    def del_unuseful_coulumns(self,train, validation, test):    
        #保留有用的columns 删除无用的columns #unit_id是index
        useful_columns = ['cycle'] + self.sensor_features + ['HI']
        train = train.loc[:,useful_columns]
        validation = validation.loc[:,useful_columns]        
        test = test.loc[:,useful_columns]
        return train, validation, test
    

    def normalization(self,train, validation, test):          
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
              
        #train.to_csv(self.root_path + '/{}/train_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        #validation.to_csv(self.root_path+ '/{}/validation_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        test.to_csv(self.root_path + 'results/{}/test_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        return train, validation, test

    
    def process(self):
        self.loader_engine()
        self.add_columns_name()
        self.labeling(piecewise_point=125)
        self.data.set_index(["unit_id"],inplace=True,drop=True)
        train, validation, test = self.part_data()
        train, validation, test = self.del_unuseful_coulumns(train, validation, test)
        train, validation, test = self.normalization(train, validation, test)
        return train, validation, test


class DataReaderTrajactory(Dataset):
    
    def __init__(self, 
                 root_path,
                 train, 
                 validation, 
                 test, 
                 flag='pred', size=None, features='MS', dataset_name='FD001', HI_labeling_style='HI_pw_quadratic',
                 target='HI', scale=True, inverse=False, timeenc=0, freq='15min',           
                 cols=None):

        self.HI_labeling_style = HI_labeling_style
        self.dataset_name = dataset_name
        
    
        # info
        self.root_path = root_path
        self.train = train
        self.validation = validation
        self.test = test
        
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
        3. prepare the data for encoder and decoder according to trajectory  96*14； 72*14  (self.transform_data)
        '''
        
        if self.flag =='train':
            self.data_x = self.train
        elif self.flag =='val':
            self.data_x = self.validation
        elif self.flag =='test':
            self.data_x = self.test
                
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
            enc_last_index = len_trajectory

            for i in range(enc_last_index - self.seq_len + 1):
                s_begin = i
                s_end = s_begin + self.seq_len
                data_enc_npc = temp_df.iloc[s_begin:s_end]
                array_data_enc.append(data_enc_npc)
            
            enc = enc + array_data_enc
                        
        #return enc,dec
        return enc,enc
    
                

    def __getitem__(self,index):  
        '''
        Parameters
        ----------
        index : int
                index 是__len__函数的返回值的一个范围： (0,range(self.__len__()))

        Returns
        -------
        seq_x : numpy array (or Tensor?)     96*14
        seq_y : numpy array (or Tensor?)     72*14
        
                # 目标：调用一次输入一组time window，也就是一组 enc_input(seq_x), dec_input(seq_y) 
                
        # __getitem__函数会被__iter__和__next__反复迭代，直到取完所有数据（迭代次数就是__len__函数的返回值）
        # index 是__len__函数的返回值的一个范围： (0,range(self.__len__()))
        # 根据自己的需要修改 __getitem__和__len__，甚至还可以修改__collate_fn__
        # 注意这个函数最后的返回值的type要是numpy array

        '''
        # seq_x = self.all_seq_x[index]
        # seq_y = self.all_seq_y[index]
        
        #删去cycle一列，并且将它转变为numpy array
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
    


