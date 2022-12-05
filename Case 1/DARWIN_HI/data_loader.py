# -*- coding: utf-8 -*-
##DARWIN  HI
import random
import math
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
                  pred_len,
                  seq_len,
                  HI_labeling_style,
                  dataset_name='FD001',                
                  **kwargs):
        
        self.root_path = root_path
        self.data_path = data_path
        self.data = pd.DataFrame([])
        self.dataset_name = dataset_name
        self.HI_labeling_style = HI_labeling_style
        self.pred_len = pred_len
        self.seq_len = seq_len
    
    # data load
    def loader_engine(self, **kwargs):
        path = os.path.join(self.data_path,'{}/train_{}.csv'.format(self.dataset_name,self.dataset_name))
        self.data = pd.read_csv( path ,header=None, **kwargs)
        self.data_test = pd.read_csv( self.data_path +'{}/test_{}.csv'.format(self.dataset_name,self.dataset_name) ,header=None, **kwargs)
        self.data_test_RUL =  pd.read_csv(self.data_path +'{}/RUL_{}.csv'.format(self.dataset_name,self.dataset_name),header=None, **kwargs)

    def add_columns_name(self):
        sensor_columns = ["sensor {}".format(s) for s in range(1,22)]
        info_columns = ['unit_id','cycle']
        settings_columns = ['setting 1', 'setting 2', 'setting 3']
        self.data.columns = info_columns + settings_columns + sensor_columns
        self.data_test.columns = info_columns + settings_columns + sensor_columns
        self.data_test_RUL.columns = ['RUL']
        self.data_test_RUL['unit_id'] = [i for i in range(1,len(self.data_test_RUL)+1,1)]
        self.data_test_RUL.set_index('unit_id',inplace=True,drop=True)
            
        
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
        self.data.drop(['HI_linear','HI_pw_linear','HI_quadratic','HI_pw_quadratic'],axis=1,inplace=True)


        ###for test
        # self.data_test.reset_index(inplace=True,drop=True)
        RUL_dict = self.data_test_RUL.to_dict()
        self.data_test['RUL_test'] =self.data_test['unit_id'].map(RUL_dict['RUL'])

        maxT_dict_train = self.data_test.groupby('unit_id')['cycle'].max().to_dict()
        self.data_test['maxT'] = self.data_test['unit_id'].map(maxT_dict_train)

        self.data_test['RUL'] = self.data_test['RUL_test'] + self.data_test['maxT'] - self.data_test['cycle']
        max_RUL_test = self.data_test.groupby('unit_id')['RUL'].max().to_dict()
        self.data_test['maxRUL'] = self.data_test['unit_id'].map(max_RUL_test)
    
        #线性
        self.data_test['HI_linear'] = 1 - self.data_test['cycle']/self.data_test['maxRUL']
        #piece_wise linear
        FPT = self.data_test['maxRUL'] - piecewise_point + 1  #+1不然无法对齐
        self.data_test['HI_pw_linear'] =  self.data_test['cycle']/ (FPT - self.data_test['maxRUL'] ) + self.data_test['maxRUL']/(self.data_test['maxRUL'] - FPT)
        filter_piece_wise = (self.data_test['cycle'] <= FPT)
        self.data_test.loc[filter_piece_wise,['HI_pw_linear']] = 1
        #quadratic
        self.data_test['HI_quadratic'] = 1 -((self.data_test['cycle']*self.data_test['cycle'])/(self.data_test['maxRUL']**2))
        #piece_wise quadratic
        self.data_test['HI_pw_quadratic'] = 1 -(((self.data_test['cycle']-FPT)**2)/(piecewise_point**2))
        filter_piece_wise = (self.data_test['cycle'] <= FPT)
        self.data_test.loc[filter_piece_wise,['HI_pw_quadratic']] = 1
        
        self.data_test['HI'] = self.data_test[self.HI_labeling_style]
        self.data_test.drop(['RUL_test','maxT','HI_linear','HI_pw_linear','HI_quadratic','HI_pw_quadratic'],axis=1,inplace=True)

    
    
    def process(self):
        self.loader_engine()
        self.add_columns_name()
        self.labeling(piecewise_point=125)
        self.data.set_index(['unit_id'],inplace=True,drop=True)
        self.data_test.set_index(['unit_id'],inplace=True,drop=True)
        train_data, test_data = self.data, self.data_test
        return train_data, test_data


class DataReaderTrajactory(Dataset):
    
    def __init__(self, root_path,rate_data, train_data,test_data, sensor_features, is_padding, data_augmentation,is_descrsing, normal_style, synthetic_data_path,
                 HI_labeling_style,flag='pred', size=None, 
                 features='MS', dataset_name='FD001', 
                 target='HI', scale=True, inverse=False, timeenc=0, freq='15min',           
                 cols=None):

        self.HI_labeling_style = HI_labeling_style
        self.dataset_name = dataset_name
        self.sensor_features = sensor_features     
        self.train_data = train_data
        self.test_data = test_data
        self.root_path = root_path
        self.rate_data = rate_data
      
        # init
        assert flag in ['train', 'test_window','test_whole', 'val']
        type_map = {'train':0, 'val':1, 'test_whole':2,'test_window':3}
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
        

        self.is_padding = is_padding
        self.data_augmentation = data_augmentation
        self.is_descrsing = is_descrsing
        self.normal_style = normal_style
        self.synthetic_data_path = synthetic_data_path


        self.__read_data__()  # data preprocessing
        
        
    def back_padding_RtF(self,data):
        '''
        1.padding in the end of each trajectory with lp lengnth data (pred_len)
        2.train data is RtF, so back padding with 0;
        3.(1)train_test split (2)normalization (3)padding;  in case of influencing the normalization process
        '''
        data_new = pd.DataFrame([],columns=data.columns)       

        for unit_index in data.index.to_series().unique():         
            unit_df = pd.DataFrame(data.loc[unit_index])               
            padding_data = pd.DataFrame(data=np.full(shape=[self.pred_len,data.shape[1]],fill_value=0),columns=data.columns)

            temp_new = pd.concat([unit_df,padding_data])
            temp_new['cycle'] = [i for i in range(1,len(temp_new)+1)]
            temp_new.index = [unit_index]*len(temp_new)
            temp_new['maxRUL'] = [unit_df['maxRUL'].max()]*(len(temp_new))  

            data_new = pd.concat((data_new,temp_new),axis=0)
        return data_new


    def HI_UtD(self,data,piecewise_point):
        '''
        1.test data is UtD, so we should construct the HI for them 
        note: no ag data in test
        '''        
        data_new = pd.DataFrame([],columns=data.columns)    
        for unit_index in data.index.to_series().unique():

            unit_df = pd.DataFrame(data.loc[unit_index])  
            padding_data = pd.DataFrame(data=np.full(shape=[self.pred_len,data.shape[1]],fill_value=0),columns=data.columns) #先用0构造
            temp_new = pd.concat([unit_df,padding_data])
            temp_new['cycle'] = [(i+unit_df['cycle'].min()) for i in range(0,len(temp_new))]
            temp_new['maxRUL'] = [unit_df['maxRUL'].max()]*(len(temp_new))
            temp_new['unit_id'] = [unit_index]*len(temp_new)

            #重新算HI
            FPT = temp_new['maxRUL'] - piecewise_point  + 1                                    
            if self.HI_labeling_style == 'HI_linear':
                temp_new['HI'] = 1 - temp_new['cycle']/temp_new['maxRUL']
            elif self.HI_labeling_style == 'HI_pw_linear':
                temp_new['HI'] =  temp_new['cycle']/ (FPT - temp_new['maxRUL'] ) + temp_new['maxRUL']/(temp_new['maxRUL'] - FPT)
                filter_piece_wise = (temp_new['cycle'] <= FPT)
                temp_new.loc[filter_piece_wise,['HI']] = 1
            elif self.HI_labeling_style == 'HI_quadratic':
                temp_new['HI'] = 1 -((temp_new['cycle']*temp_new['cycle'])/(temp_new['maxRUL']**2))
            elif self.HI_labeling_style == 'HI_pw_quadratic':
                temp_new['HI'] = 1 -(((temp_new['cycle']-FPT)**2)/(piecewise_point**2))
                filter_piece_wise = (temp_new['cycle'] <= FPT)
                temp_new.loc[filter_piece_wise,['HI']] = 1


            filter_neg_value = (temp_new['HI'] < 0)
            temp_new.loc[filter_neg_value,['HI']] = 0            

            temp_new.drop('maxRUL',inplace=True,axis=1)                                              
            data_new = pd.concat([data_new,temp_new],axis=0)

        data_new.set_index('unit_id',inplace=True,drop=True)
        data = data_new

        return data
        
              
    
    def __read_data__(self):      
        '''
        1. spilt for data,val,test according to trajectory
        2. normalization
        3. prepare the data for encoder and decoder according to trajectory  96*14； 72*14  (self.transform_data)
        '''
        
        # load data
        if self.data_augmentation == True:
            df_raw = self.train_data
            df_ag = pd.read_csv(self.synthetic_data_path,header=0,index_col=['unit_id'])    

            #####data availability
            if self.rate_data != 1:
                unique_raw = df_raw.index.to_series().unique()
                num_raw = len(unique_raw)
                unique_ag = df_ag.index.to_series().unique()
                num_ag = len(unique_ag)
                raw_unit_used = random.sample(unique_raw.tolist(),math.ceil(num_raw*self.rate_data))
                ag_unit_used = random.sample(unique_ag.tolist(),math.ceil(num_ag*self.rate_data))
                df_raw = df_raw.loc[raw_unit_used]
                df_ag = df_ag.loc[ag_unit_used]
            
        
        else:
            df_raw = self.train_data

          
        ## test 
        test = self.test_data

      
        train_turbines = np.arange(len(df_raw.index.to_series().unique()))
        train_turbines, validation_turbines = train_test_split(train_turbines, test_size=0.3,random_state = 1334) 
        idx_train = df_raw.index.to_series().unique()[train_turbines]
        idx_validation = df_raw.index.to_series().unique()[validation_turbines]
        train = df_raw.loc[idx_train]
        validation = df_raw.loc[idx_validation]
            
            
        if self.data_augmentation == True:
            train_turbines_ag = np.arange(len(df_ag.index.to_series().unique()))  
            train_turbines_ag, validation_turbines_ag = train_test_split(train_turbines_ag, test_size=0.3,random_state = 1334)
            idx_train_ag = df_ag.index.to_series().unique()[train_turbines_ag]
            idx_validation_ag = df_ag.index.to_series().unique()[validation_turbines_ag]
            train_ag = df_ag.loc[idx_train_ag]
            validation_ag = df_ag.loc[idx_validation_ag]
            
                
        # normalization
        if self.normal_style == 'StandardScaler':
            self.scaler = StandardScaler()
        elif self.norma_style == 'MinMaxScale':
            self.scaler = MinMaxScaler()
            
            
        train_normalize = train.copy()
        validation_normalize = validation.copy()
        test_normalize = test.copy()
        

        self.scaler.fit(train.loc[:,self.sensor_features].values) # use the mean and variance in data dataset rather the whole data
        train_normalize.loc[:,self.sensor_features] = self.scaler.transform(train.loc[:,self.sensor_features].values)  #do not normal cycle and HI
        validation_normalize.loc[:,self.sensor_features]= self.scaler.transform(validation.loc[:,self.sensor_features].values)
        test_normalize.loc[:,self.sensor_features] = self.scaler.transform(test.loc[:,self.sensor_features].values)

             
        train =  train_normalize.copy()
        validation = validation_normalize.copy()
        test = test_normalize.copy()
        del (train_normalize,validation_normalize,test_normalize) 
        
        
        if self.is_descrsing==True:
            sensor_in = ['sensor 2','sensor 3','sensor 4','sensor 8','sensor 9','sensor 11','sensor 13','sensor 15','sensor 17']
            sensor_de = ['sensor 7','sensor 12','sensor 14','sensor 20','sensor 21']
            train[sensor_in] = 1 - train[sensor_in]
            validation[sensor_in] = 1 - validation[sensor_in]
            test[sensor_in] = 1 - test[sensor_in]
            
                       
        #padding
        if self.is_padding==True:
            train = self.back_padding_RtF(train)
            validation = self.back_padding_RtF(validation)     
        
        #get the HI for test data
        test = self.HI_UtD(test,piecewise_point=125) 
                         
        # useful_columns = ['cycle'] + self.sensor_features + ['HI']
        useful_columns =  self.sensor_features + ['HI']
        train = train.loc[:,useful_columns]
        validation = validation.loc[:,useful_columns]        
        test = test.loc[:,useful_columns]
        
        
        if self.data_augmentation == True:
            train = pd.concat([train,train_ag],axis=0)
            validation = pd.concat([validation,validation_ag],axis=0)
            
        
        ###test get the last window and the whole windows 
        '''note: since we have alreadt padded the data, so the lenght should be greater then seq_len+pred_len'''
        true_data,true_window_data = pd.DataFrame([]),pd.DataFrame([])
        for unit_index in (test.index.to_series().unique()):
            #get the whole trajectory (index)
            trajectory_df = pd.DataFrame(test.loc[unit_index])

            
            if len(trajectory_df) >= (self.seq_len +self.pred_len) :
                #whole windows in test
                true_data = pd.concat([true_data,trajectory_df])
                
                #the last window in test
                temp_last_new = trajectory_df.iloc[(-self.seq_len-self.pred_len):,:]  
                true_window_data = pd.concat([true_window_data,temp_last_new])
            
            ##pad the short lenghth of test data
            else:   
                padding_data = pd.DataFrame(data=np.full(shape=[-len(trajectory_df)+self.seq_len+self.pred_len,trajectory_df.shape[1]],fill_value=1),columns=trajectory_df.columns)
                temp_last_new = pd.concat([padding_data,trajectory_df])
                temp_last_new['unit_id'] = [unit_index]*len(temp_last_new)
                temp_last_new.set_index(['unit_id'],inplace=True,drop=True)
                true_window_data = pd.concat([true_window_data,temp_last_new])
                
                true_data = pd.concat([true_data,temp_last_new]) 
        
        
        # whole windows in test --- for prediction
        #true_data.to_csv(self.root_path +'/results/{}/true_data_{}.csv'.format(self.dataset_name,self.HI_labeling_style),header=True,index=True)
        # only the last window --- for test
        #true_window_data.to_csv(self.root_path +'/results/{}/true_window_{}.csv'.format(self.dataset_name,self.HI_labeling_style),header=True,index=True)   
        
        test_whole = true_data
        test_window = true_window_data
        
        train.to_csv(self.root_path + '/results/{}/train_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        validation.to_csv(self.root_path+ '/results/{}/validation_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        test_window.to_csv(self.root_path + '/results/{}/test_window_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        test_whole.to_csv(self.root_path + '/results/{}/test_whole_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
  

        if self.flag =='train':
            self.data_x = train
        elif self.flag =='val':
            self.data_x = validation
        elif self.flag =='test_window':
            self.data_x = test_window
        elif self.flag =='test_whole':
            self.data_x = test_whole           
            
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
        
              
        for index in (self.data_x.index.to_series().unique()): 
            
            #get the whole trajectory (index)
            temp_df = pd.DataFrame(self.data_x.loc[index])             
             
            # Loop through the data in the object (index) trajectory
            data_enc_npc, data_dec_npc, array_data_enc, array_data_dec = [],[],[],[]
            len_trajectory = len(temp_df)
            
            
            enc_last_index = len_trajectory - self.pred_len
            

            for i in range(enc_last_index - self.seq_len + 1):
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len 
                r_end = r_begin + self.label_len + self.pred_len 

                data_enc_npc = temp_df.iloc[s_begin:s_end]
                data_dec_npc = temp_df.iloc[r_begin:r_end]
       
                array_data_enc.append(data_enc_npc)
                array_data_dec.append(data_dec_npc)
        
            enc = enc + array_data_enc
            dec = dec + array_data_dec

        return enc,dec
    
                

    def __getitem__(self,index):  
        '''
        Parameters
        ----------
        index : int
                index: (0,range(self.__len__()))

        Returns
        -------
        seq_x : numpy array (or Tensor?)     96*14
        seq_y : numpy array (or Tensor?)     72*14
        
                # target：return one set of time window: enc_input(seq_x), dec_input(seq_y) 
                
        # __getitem__ will be repeated iteratition by __iter__ and __next__, until all the data are gained
        # the total number of iteration is the return value of __len__  
        # index is a  range of __len__: (0,range(self.__len__()))
        # according to your case, you could revise the  __getitem__, __len__ and __collate_fn__ function
        # note: the return value of __getitem__ is numpy array type 

        '''
        # seq_x = self.all_seq_x[index]
        # seq_y = self.all_seq_y[index]
        
        #trun them to numpy array type
        seq_x = self.all_seq_x[index].values
        seq_y = self.all_seq_y[index].values
        
        #return seq_x, seq_y
        ## TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found
        return seq_x.astype(np.float32),seq_y.astype(np.float32)  
        
        
    def __len__(self):
        '''   
        Returns: len(self.all_seq_x) 
        TYPE: int
              #seq_len * enc_in （96*14) nunber of training time windows
        '''
        
        return len(self.all_seq_x)   
    
    def inverse_transform(self, data):
        
        return self.scaler.inverse_transform(data)
    


# class Dataset_Pred_Trajectory(Dataset):
