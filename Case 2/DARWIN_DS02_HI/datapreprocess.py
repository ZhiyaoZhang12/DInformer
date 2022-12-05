#### DARWIN  DS02 hi
import os, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from scipy.stats import reciprocal, randint, uniform
from math import ceil
import pandas as pd
import numpy as np
import math
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import joblib
import h5py
import random
from utils.lttb import LTTB  #下采样时序数据



class DataPreproserse():  
    
    '''
    1.load_data                            ：index ['unit']
    2.del unuseful columns (0 values)      ：注意保留unit,cycle
    3.normalization                        ：train数据的max,min,mean,var给test数据normalization
    4.one-hot coding for discrete features : Fc, hs (hs不需要再one hot了，可以直接用)
    5.HI labeling                          ：注意在这里删除了t,tf,Tn 和 cycle,RUL (先one-hot再HI,可以保证HI在最后一列)
    6.RtF pading                           ：这个一定要在normalization之后,以免影响max,min,mean,var
    7.get window                           ：pars:seq_len,label_len,pred_len 决定了窗口的大小和多少
    8.train validation split               ：注意不要按照unit划分train和vali,而是时间窗打乱了之后再区分，因为每个unit工况不同，这样train数据里就包含了更多可能的工况
    9.No save data                         ：time window数据量很大,超过10g,且不同的sl,ll,pl就有不同的时间窗组合,这个数据不是完全相同的
    
    return train validation test
    (train:2,5,10,16,18,20; test:11,14,15)
    '''
    def __init__(self,
                 root_path,
                 dataset_name,
                 data_path,
                 validation_split,
                 normal_style,
                 down_sampling,
                 down_sampling_rate,
                 ag_data_len,
                 ag_seq_len,
                 stride,
                 seq_len,
                 label_len,
                 pred_len,
                 is_padding,
                 data_augmentation,
                 rate_data,
                 synthetic_data_path,
                  ):
    
        self.root_path = root_path
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.validation_split = validation_split
        self.normal_style = normal_style
        self.down_sampling = down_sampling
        self.down_sampling_rate = down_sampling_rate
        self.ag_data_len = ag_data_len
        self.ag_seq_len = ag_seq_len
        self.stride = stride
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.is_padding = is_padding
        self.data_augmentation = data_augmentation  #(暂时不考虑ag,先跑通程序)
        self.rate_data = rate_data  #暂不考虑
        self.synthetic_data_path=synthetic_data_path  #暂不考虑
        
        self.type_1 = ['FD001', 'FD003','DS02']   #暂时归入type1
        self.type_2 = ['FD002', 'FD004','PHM08']
        
    
    def load_engine(self):
        if self.down_sampling=='False': ##daown_sampling in ['False','simple_sampling','LTTB_sampling']
            hdf = h5py.File(self.data_path + 'N-CMAPSS_DS02-006.h5', "r")
        elif self.down_sampling in ['simple_sampling','LTTB_sampling']:
            hdf = h5py.File(self.data_path + 'DS02_{}_rate{}.h5'.format(self.down_sampling,self.down_sampling_rate), "r")
        sampled_train = np.array(hdf.get('df_dev'))
        sampled_test = np.array(hdf.get('df_test'))
        columns_name = np.array(hdf.get('columns_name'))
        columns_name = list(np.array(columns_name, dtype='U20'))
        hdf.close()

        sampled_train = pd.DataFrame(data=sampled_train,columns=columns_name)
        sampled_test = pd.DataFrame(data=sampled_test,columns=columns_name)                  
        
        sampled_train.set_index(['unit'],inplace=True,drop=True)
        sampled_test.set_index(['unit'],inplace=True,drop=True)
        
        
        ###11.20删除unit 14 (工况太复杂，太短了)
        sampled_test.drop(index=14.0,inplace=True)
        print('unit in test:', sampled_test.index.to_series().unique(),'shape:',sampled_train.shape)
        print('unit in train:', sampled_train.index.to_series().unique(),'shape',sampled_test.shape)       
        
        print('length of each dev unit after down sampling (rate: {}):'.format(self.down_sampling_rate),\
                              [len(sampled_train.loc[uni_index]) for uni_index in sampled_train.index.to_series().unique()])
        
        print('length of each test unit after down sampling (rate: {}):'.format(self.down_sampling_rate),\
                              [len(sampled_test.loc[uni_index]) for uni_index in sampled_test.index.to_series().unique()])
        

        return sampled_train, sampled_test
    
    
    def del_unuseful_features(self,data):
        ##constant value 0, delect them
        unuseful_features = ["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", \
                                      "HPC_flow_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod"] #, "cycle", "unit"]    #unit and cycle is useful for data preprocess  
        return data.drop(unuseful_features,axis=1)
    
    
    def normalization(self,df_dev, df_test): 
        categoricalFeatures = ["Fc", "hs"]   #one-hot encoder
        not_normal_columns = categoricalFeatures + ['RUL'] + ['cycle','unit']
        continuousFeatures = [x for x in df_dev.columns.tolist() if x not in not_normal_columns] #01 normalization

        df_dev_normalize = df_dev.copy()
        df_test_normalize = df_test.copy()

        if self.dataset_name in self.type_1:
            if self.normal_style == 'StandardScaler':
                scaler = StandardScaler().fit(df_dev[continuousFeatures])
            elif self.normal_style == 'MinMaxScaler':
                scaler = MinMaxScaler().fit(df_dev[continuousFeatures])

            df_dev_normalize[continuousFeatures] = scaler.transform(df_dev[continuousFeatures])
            df_test_normalize[continuousFeatures] = scaler.transform(df_test[continuousFeatures])
            
            
        elif self.dataset_name in self.type_2:   
            #给他们聚类['OP']   ###多少种工况？16？
            kmeans = KMeans(n_clusters=15, random_state=0).fit(df_dev[self.operating_settings])
            df_dev['OP'] = kmeans.labels_
            df_test['OP'] = kmeans.predict(df_test[self.operating_settings])

            if len(df_dev):
                df_dev_normalize = df_dev.copy()
                df_test_normalize = df_test.copy()

            gb = df_dev.groupby('OP')[continuousFeatures]

            d = {}
            for x in gb.groups:
                if self.normal_style == 'StandardScaler':
                    d["scaler_{0}".format(x)] = StandardScaler().fit(gb.get_group(x))
                elif self.normal_style == 'MinMaxScaler':
                    d["scaler_{0}".format(x)] = MinMaxScaler().fit(gb.get_group(x))

                df_dev_normalize.loc[df_dev_normalize['OP'] == x, continuousFeatures] = d["scaler_{0}".format(x)].transform(
                    df_dev.loc[df_dev['OP'] == x, continuousFeatures])
                df_test_normalize.loc[df_test_normalize['OP'] == x, continuousFeatures] = d["scaler_{0}".format(x)].transform(
                    df_test.loc[df_test['OP'] == x, continuousFeatures])
         
        df_dev = df_dev_normalize.copy()
        df_test = df_test_normalize.copy()
        del df_dev_normalize, df_test_normalize          
        return df_dev, df_test
    
       
    def onehot_coding(self,df_dev,df_test):  
        categoricalFeatures = ["Fc", "hs"]   #one-hot encoder  Fc(1,2,3) three fight classes; hs:(0,1)health stages  
        one_hot = LabelBinarizer()
        one_hot.fit([1.0,2.0,3.0])
        
        new_columns = ["Fc{}".format(s) for s in range(1,4)]
        new_data_dev = one_hot.transform(df_dev['Fc'])
        new_data_test = one_hot.transform(df_test['Fc'])
        
        new_data_dev = pd.DataFrame(data=new_data_dev,columns=new_columns,index=df_dev.index)
        new_data_test = pd.DataFrame(data=new_data_test,columns=new_columns,index=df_test.index)
        
        df_dev.drop(['Fc'],inplace=True,axis=1)
        df_test.drop(['Fc'],inplace=True,axis=1)
        df_dev = pd.concat([df_dev,new_data_dev],axis=1)
        df_test = pd.concat([df_test,new_data_test],axis=1)
        
        del new_data_dev, new_data_test
        return df_dev, df_test
    
    
    def HI_labeling(self,data):  
        data_new = pd.DataFrame([])
        for index_data in (data.index.to_series().unique()):  
            temp_data = data.loc[index_data]       
            #temp_data['t'] = [10*(i+1) for i in range(temp_data.shape[0])]  #sampling rate is 0.1hz, that is 10s duration  
            temp_data['t'] = [(10/self.down_sampling_rate)*(i+1) for i in range(temp_data.shape[0])]  #其实除不除rate不影响HI值，因为分子分母同时约掉了，只是影响了t而已
            filter1 = temp_data['hs']==0
            tf = temp_data.loc[filter1,'t'].min()
            Tn = temp_data.loc[filter1,'t'].max() 
            temp_data['tf'] = [tf]*temp_data.shape[0]
            temp_data['Tn'] = [Tn]*temp_data.shape[0] 
            temp_data['HI'] = 1 - ((temp_data['t']-temp_data['tf'])**2/(temp_data['Tn']-temp_data['tf'])**2)
            
            filterhs = temp_data['hs']==1
            temp_data.loc[filterhs,'HI'] = 1       
            #print('Unit: ' + str(index_data) + ' - tf: ', tf, ' - Number of flight cyles (t_{EOF}): ',Tn)
            data_new = pd.concat([data_new,temp_data],axis=0)
        
        data_new.drop(['tf','t','Tn','RUL','cycle'],inplace=True,axis=1) #'RUL','HI'   #'cycle',后面需要用到cycle  ---- 其实也可以考虑cycle也作为特征
        return data_new
    

    
    def back_padding_RtF(self,data):
        '''
        1.padding in the end of each trajectory with lp lengnth data (pred_len)
        2.train data is RtF, so back padding with 0;
        3.normalization before padding, in case of influencing the normalization process
        '''
        data_new = pd.DataFrame([],columns=data.columns)       

        for unit_index in data.index.to_series().unique():         
            unit_df = pd.DataFrame(data.loc[unit_index])               
            padding_data = pd.DataFrame(data=np.full(shape=[self.pred_len,data.shape[1]],fill_value=0),columns=data.columns)

            temp_new = pd.concat([unit_df,padding_data])
            temp_new.index = [unit_index]*len(temp_new)
            data_new = pd.concat((data_new,temp_new),axis=0)
        return data_new
    
    
    def add_data_augmantation_data(self,df_dev):
        if self.data_augmentation==True:
            f = h5py.File('Case/DARWIN/DARWIN_DS02_AG/results/Synthetics_DS02_{}_rate{}_len{}_{}.h5'.format(self.down_sampling,\
                                                                                self.down_sampling_rate,self.ag_data_len,self.ag_seq_len), "r")     
            columns_name = ['unit'] + df_dev.columns.to_list()
            ag_data_df = pd.DataFrame(data=np.array(f.get('ag_data')),columns=columns_name)
            ag_data_df.set_index(['unit'],inplace=True,drop=True)
            
            print('*****************************')
            print(len(ag_data_df.index.to_series().unique()))
            print(ag_data_df.index.to_series().unique())
            
            df_dev = pd.concat([df_dev,ag_data_df],axis=0) 
            f.close()    
            print('length of dev units after data augumentation (rate: {}):'.format(self.down_sampling_rate),\
                              [len(df_dev.loc[uni_index]) for uni_index in df_dev.index.to_series().unique()])
            
        return df_dev
    
    
    def transform_data2window(self,data):
        ### enc, dec for save the precessed data(time window) 96*14 
        enc,dec = [],[]
        #Loop through each unit
        num_units = len(data.index.to_series().unique())
        print('length of each unit after down sampling (rate: {}):'.format(self.down_sampling_rate),[len(data.loc[uni_index]) for uni_index in data.index.to_series().unique()])
        
        for index in (data.index.to_series().unique()): 
            #get the whole trajectory (index)
            temp_df = pd.DataFrame(data.loc[index])             

            # Loop through the data in the object (index) trajectory
            data_enc_npc, data_dec_npc, array_data_enc, array_data_dec = [],[],[],[]
            len_trajectory = len(temp_df)
            enc_last_index = len_trajectory - self.pred_len
            
            i = 0
            while i+self.seq_len+self.pred_len <= len_trajectory:
            #for i in range(enc_last_index - self.seq_len + 1):
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len 
                r_end = r_begin + self.label_len + self.pred_len 

                data_enc_npc = temp_df.iloc[s_begin:s_end]
                data_dec_npc = temp_df.iloc[r_begin:r_end]
                array_data_enc.append(data_enc_npc)
                array_data_dec.append(data_dec_npc)
                
                i += self.stride

            enc = enc + array_data_enc
            dec = dec + array_data_dec
        return enc, dec
    
    
    def split_train_vali(self, dev_enc, dev_dec):        
        #X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state=random_state)
        #借用这个函数，assume enc是features,dec是labels  这样可以保障enc和dec是成对split的
        train_enc, val_enc, train_dec, val_dec = train_test_split(dev_enc, dev_dec,test_size=self.validation_split,shuffle=True,random_state=110)
        return train_enc, val_enc, train_dec, val_dec

    
    def process(self):
        df_dev, df_test = self.load_engine()
        df_dev, df_test = self.del_unuseful_features(df_dev), self.del_unuseful_features(df_test)
        df_dev, df_test = self.normalization(df_dev,df_test)
        df_dev, df_test = self.onehot_coding(df_dev,df_test) 
        df_dev, df_test = self.HI_labeling(df_dev),self.HI_labeling(df_test)     
        if self.is_padding:
            df_dev = self.back_padding_RtF(df_dev)
        df_test = self.back_padding_RtF(df_test)  ###test数据是一定padding的，为了检验模型效果   

        print('*************************columns used:*************************')    
        print(df_dev.columns)
        print('\n')
                          
        df_dev = self.add_data_augmantation_data(df_dev)                  
        dev_enc, dev_dec = self.transform_data2window(df_dev)
        test_enc, test_dec = self.transform_data2window(df_test)
        train_enc, val_enc, train_dec, val_dec = self.split_train_vali(dev_enc,dev_dec)
        
        print('No. of train samples:',len(train_enc))
        print('No. of val samples:',len(val_enc))
        print('No. of test samples:',len(test_enc))
    
        return  train_enc, train_dec, val_enc, val_dec, test_enc, test_dec

        
if __name__ == "__main__":
    datapreprocess = DataPreproserse(
                        root_path='Case/DRAWIN/DARWIN_DS02/',
                        dataset_name='DS02',
                        validation_split = 0.3,
                        normal_style='StandardScaler',
                        down_sampling='LTTB_sampling',
                        down_sampling_rate=6,
                        stride = 2,
                        seq_len=64,
                        label_len=32,
                        pred_len=64,
                        is_padding=True,
                        data_augmentation=True,
                        rate_data=0.6,
                        synthetic_data_path='Case/DARWIN/DARWIN_AG/results/FD001/Synthetic_HI_pw_quadratic_StandardScaler.csv',
                        )
    train_enc, train_dec, val_enc, val_dec, test_enc, test_dec = datapreprocess.process()
