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
                  pred_len,
                  seq_len,
                  HI_labeling_style,
                  dataset_name='FD001',                
                  **kwargs):
        
        self.root_path = root_path
        self.data = pd.DataFrame([])
        self.dataset_name = dataset_name
        self.HI_labeling_style = HI_labeling_style
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.__process__()
    
    # data load
    def loader_engine(self, **kwargs):
        path = os.path.join(self.root_path,'{}/train_{}.csv'.format(self.dataset_name,self.dataset_name))
        self.data = pd.read_csv( path ,header=None, **kwargs)
        self.data_test = pd.read_csv( self.root_path +'{}/test_{}.csv'.format(self.dataset_name,self.dataset_name) ,header=None, **kwargs)
        self.data_test_RUL =  pd.read_csv(self.root_path +'{}/RUL_{}.csv'.format(self.dataset_name,self.dataset_name),header=None, **kwargs)

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
    
        #linear
        self.data_test['HI_linear'] = 1 - self.data_test['cycle']/self.data_test['maxRUL']
        #piece_wise linear
        FPT = self.data_test['maxRUL'] - piecewise_point + 1 
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

    def __process__(self):
        self.loader_engine()
        self.add_columns_name()
        self.labeling(piecewise_point=125)
        self.data.to_csv(self.root_path + '/{}/train_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=False)
        self.data_test.to_csv(self.root_path + '/{}/test_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=False)

class DataReaderTrajactory(Dataset):
    
    def __init__(self, root_path,sensor_features, is_padding, data_augmentation,is_descrsing, normal_style, synthetic_data_path,
                 HI_labeling_style,flag='pred', size=None, 
                 features='MS', dataset_name='FD001', 
                 target='HI', scale=True, inverse=False, timeenc=0, freq='15min',           
                 cols=None):

        self.HI_labeling_style = HI_labeling_style
        self.dataset_name = dataset_name
        self.sensor_features = sensor_features
        
        # info
        self.root_path = root_path
        self.data_path = '{}/train_{}.csv'.format(self.dataset_name,self.HI_labeling_style)
        self.data_path_test = '{}/test_{}.csv'.format(self.dataset_name,self.HI_labeling_style)

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
              
    
    def __read_data__(self):
          
        '''
        1. spilt for data,val,test according to trajectory
        2. normalization
        3. prepare the data for encoder and decoder according to trajectory  96*14； 72*14  (self.transform_data)
        '''
        
        # load data
        if self.data_augmentation == True:
            df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path),header=0)
            df_raw.set_index(['unit_id'],inplace=True,drop=True)
            df_raw_fake = pd.read_csv(self.synthetic_data_path,header=0)   
            df_raw_fake.set_index(['unit_id'],inplace=True,drop=True)            
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path),header=0)             
            df_raw.set_index(['unit_id'],inplace=True,drop=True)
        
        ## test 
        test = pd.read_csv(os.path.join(self.root_path,self.data_path_test),header=0, index_col=['unit_id'])           
        train_turbines = np.arange(len(df_raw.index.to_series().unique()))
        train_turbines, validation_turbines = train_test_split(train_turbines, test_size=0.3,random_state = 1334) 
        idx_train = df_raw.index.to_series().unique()[train_turbines]
        idx_validation = df_raw.index.to_series().unique()[validation_turbines]
        train = df_raw.loc[idx_train]
        validation = df_raw.loc[idx_validation]
            
        if self.data_augmentation == True:
            train_turbines_fake = np.arange(len(df_raw_fake.index.to_series().unique()))  
            train_turbines_fake, validation_turbines_fake = train_test_split(train_turbines_fake, test_size=0.3,random_state = 1334)  
            idx_train_fake = df_raw_fake.index.to_series().unique()[train_turbines_fake]
            idx_validation_fake = df_raw_fake.index.to_series().unique()[validation_turbines_fake]
            train_fake = df_raw_fake.loc[idx_train_fake]
            validation_fake = df_raw_fake.loc[idx_validation_fake]
            
                
        # normalization
        if self.normal_style == 'StandardScaler':
            self.scaler = StandardScaler()
        elif self.norma_style == 'MinMaxScale':
            self.scaler = MinMaxScaler()
            
        train_normalize = train.copy()
        validation_normalize = validation.copy()
        test_normalize = test.copy()
        

        self.scaler.fit(train.loc[:,self.sensor_features].values)     # use the mean and variance in data dataset, do not use the mean and variance of the whole data
        train_normalize.loc[:,self.sensor_features] = self.scaler.transform(train.loc[:,self.sensor_features].values)  #do not standalise fot cycle and HI 
        validation_normalize.loc[:,self.sensor_features]= self.scaler.transform(validation.loc[:,self.sensor_features].values)
        test_normalize.loc[:,self.sensor_features] = self.scaler.transform(test.loc[:,self.sensor_features].values)

             
        train =  train_normalize.copy()
        validation = validation_normalize.copy()
        test = test_normalize.copy()
        del (train_normalize,validation_normalize,test_normalize) 
             
        #descreasing sensor 7,12,14,20,21
        #increasing  sensor 2,3,4,8,9,11,13,15,17
        if self.is_descrsing==True:   #default= False
            sensor_in = ['sensor 2','sensor 3','sensor 4','sensor 8','sensor 9','sensor 11','sensor 13','sensor 15','sensor 17']
            sensor_de = ['sensor 7','sensor 12','sensor 14','sensor 20','sensor 21']
            train[sensor_in] = 1 - train[sensor_in]
            validation[sensor_in] = 1 - validation[sensor_in]
            test[sensor_in] = 1 - test[sensor_in]
            
        ###padding
        def back_padding_RtF(data):
            '''
            1.pad a zero-matric (pred_len*NO.sensors) in the end of each trajectory;    
            2.train_test split then normalization，then padding;
            '''
            data_new = pd.DataFrame([],columns=data.columns) #for saving new data         
        
            for unit_index in data.index.to_series().unique():                      
                unit_df = pd.DataFrame(data.loc[unit_index])    #trajectory of each unit            
                padding_data = pd.DataFrame(data=np.full(shape=[self.pred_len,data.shape[1]],fill_value=0),columns=data.columns)

                #pad
                temp_new = pd.concat([unit_df,padding_data])
                temp_new['cycle'] = [i for i in range(1,len(temp_new)+1)]
                temp_new.index = [unit_index]*len(temp_new)
                temp_new['maxRUL'] = [unit_df['maxRUL'].max()]*(len(temp_new))                      
                data_new = pd.concat((data_new,temp_new),axis=0)
            return data_new
        
        
        def HI_UtD(data,piecewise_point):
            '''
            test data is UtD, so construct the predicted PDIs
            1.back_padding for test data, then modified the PDIs
            '''   
            data_new = pd.DataFrame([],columns=data.columns)  #for saving new data   
                   
            for unit_index in data.index.to_series().unique():
                unit_df = pd.DataFrame(data.loc[unit_index])    #trajectory of each unit  
                
                padding_data = pd.DataFrame(data=np.full(shape=[self.pred_len,data.shape[1]],fill_value=0),columns=data.columns) #use zero-matrics first
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
                                  
                
                #if the constructed PDIs is less than 0,  use 0 instead
                filter_neg_value = (temp_new['HI'] < 0)
                temp_new.loc[filter_neg_value,['HI']] = 0            
                                        
                temp_new.drop('maxRUL',inplace=True,axis=1)
                                                    
                #concat
                data_new = pd.concat([data_new,temp_new],axis=0)
                
            data_new.set_index('unit_id',inplace=True,drop=True)
            data = data_new
                    
            return data
                       
        #padding
        if self.is_padding==True:
            train = back_padding_RtF(train)
            validation = back_padding_RtF(validation)     
               
        test = HI_UtD(test,piecewise_point=125) 
                                
        #deldete unuseful columns #unit_id:index
        useful_columns =  self.sensor_features + ['HI']
        train = train.loc[:,useful_columns]
        validation = validation.loc[:,useful_columns]        
        test = test.loc[:,useful_columns]
        
        
        if self.data_augmentation == True:
            train = pd.concat([train,train_fake],axis=0)
            validation = pd.concat([validation,validation_fake],axis=0)
            
        
        ###test get the last window
        pred_data,window_data = pd.DataFrame([]),pd.DataFrame([])
        for unit_index in (test.index.to_series().unique()):

            trajectory_df = pd.DataFrame(test.loc[unit_index])

            if len(trajectory_df) >= (self.seq_len +self.pred_len) :
                temp_last_new = trajectory_df.iloc[(-self.seq_len-self.pred_len):,:]  
                window_data = pd.concat([window_data,temp_last_new])             
            else:                
                padding_data = pd.DataFrame(data=np.full(shape=[-len(trajectory_df)+self.seq_len+self.pred_len,trajectory_df.shape[1]],fill_value=1),columns=trajectory_df.columns)
                temp_last_new = pd.concat([padding_data,trajectory_df])
                temp_last_new['unit_id'] = [unit_index]*len(temp_last_new)
                temp_last_new.set_index(['unit_id'],inplace=True,drop=True)
                window_data = pd.concat([window_data,temp_last_new])       

        window_data.to_csv(self.root_path +'/{}/test_window_{}.csv'.format(self.dataset_name,self.HI_labeling_style),header=True,index=True)   

        test_whole = pred_data
        test_window = window_data
               
        train.to_csv(self.root_path + '/{}/train_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        validation.to_csv(self.root_path+ '/{}/validation_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)
        test_window.to_csv(self.root_path + '/{}/test_window_normal_{}.csv'.format(self.dataset_name,self.HI_labeling_style),index=True)

  

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
            
        ## prepare the data for encoder and decoder according to trajectory:
        self.all_seq_x, self.all_seq_y = self.transform_data()

        
    def transform_data(self):
        ### enc, dec for save the precessed data(time window) 
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
        
        seq_x = self.all_seq_x[index].values
        seq_y = self.all_seq_y[index].values
        
        #return seq_x, seq_y
        ##TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found
        return seq_x.astype(np.float32),seq_y.astype(np.float32)  
        
        
    def __len__(self):
        
        return len(self.all_seq_x)   
    
    def inverse_transform(self, data):
        
        return self.scaler.inverse_transform(data)
