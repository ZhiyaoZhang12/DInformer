#### DARWIN  DS02 down_sampling
'''
###保存h5文件  后续直接读取它即可

'''
#### DARWIN  DS02
import numpy as np
import pandas as pd
import h5py
from lttb import LTTB  #下采样时序数据


class DownSampling():  
    def __init__(self,
                 data_path,
                 down_sampling,
                 down_sampling_rate,
                  ):
    
        self.data_path = data_path
        self.down_sampling = down_sampling
        self.down_sampling_rate = down_sampling_rate

    
    def load_engine(self):
        with h5py.File(self.data_path, "r") as hdf:
            #print("Keys: {}".format(hdf.keys()))
            # Development set
            W_dev = np.array(hdf.get('W_dev'))             # W  - operative condition
            X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s - measured signal
            X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v - virtual sensors
            T_dev = np.array(hdf.get('T_dev'))             # T - engine health parameters
            Y_dev = np.array(hdf.get('Y_dev'))             # RUL - RUL label
            A_dev = np.array(hdf.get('A_dev'))             # Auxiliary - unit number u and the flight cycle number c, the flight class Fc and the health state h s

            # Test set
            W_test = np.array(hdf.get('W_test'))           # W
            X_s_test = np.array(hdf.get('X_s_test'))       # X_s
            X_v_test = np.array(hdf.get('X_v_test'))       # X_v
            T_test = np.array(hdf.get('T_test'))           # T
            Y_test = np.array(hdf.get('Y_test'))           # RUL  
            A_test = np.array(hdf.get('A_test'))           # Auxiliary

            # Varnams
            W_var = np.array(hdf.get('W_var'))
            X_s_var = np.array(hdf.get('X_s_var'))  
            X_v_var = np.array(hdf.get('X_v_var')) 
            T_var = np.array(hdf.get('T_var'))
            A_var = np.array(hdf.get('A_var'))

            # from np.array to list dtype U4/U5
            W_var = list(np.array(W_var, dtype='U20'))
            X_s_var = list(np.array(X_s_var, dtype='U20'))  
            X_v_var = list(np.array(X_v_var, dtype='U20')) 
            T_var = list(np.array(T_var, dtype='U20'))
            A_var = list(np.array(A_var, dtype='U20'))

            self.operating_settings = W_var

        dev_data = np.concatenate((W_dev, X_s_dev, X_v_dev, T_dev, A_dev, Y_dev), axis=1)
        test_data = np.concatenate((W_test, X_s_test, X_v_test, T_test, A_test, Y_test), axis=1)
        column_name = W_var + X_s_var + X_v_var + T_var + A_var
        column_name.append("RUL")

        print("dev_data shape: {}".format(dev_data.shape))
        print("test_data shape: {}".format(test_data.shape))    

        df_dev = pd.DataFrame(data=dev_data, columns=column_name)
        df_test = pd.DataFrame(data=test_data, columns=column_name)
        
        print("units in dev_data: {}".format(df_dev['unit'].unique()))
        print("units in test_data: {}".format(df_test['unit'].unique()))
        
        
        with open( "./DS02_{}_rate{}_dataset_info.txt".format(self.down_sampling,self.down_sampling_rate), "w+") as f1:
            f1.write("\ntrain shape after sampled: {}".format(df_dev.shape))
            f1.write("\ntest shape after sampled: {}".format(df_test.shape))
        
        
        df_dev.set_index(['unit'],inplace=True,drop=True)
        df_test.set_index(['unit'],inplace=True,drop=True)
        return df_dev, df_test
    
    
    def SDS_sampling(self,data):
        '''
        observation duration is 10s since the sampling rate is 0.1hz;
        if down sampling is needed, change the observation duration to 20s to sve the memory
        '''
        data_new = pd.DataFrame([])
        for index_data in (data.index.to_series().unique()):  
            temp_data = data.loc[index_data] 
            sampling_columns = [i*2 for i in range(temp_data.shape[0]//2)]
            data_new = pd.concat([data_new,temp_data.iloc[sampling_columns]],axis=0)
        data = data_new
        del data_new
        return data

    
    def LTTB_sampling(self,data):
        data_new = pd.DataFrame([])
        print(data.shape)
        for index_unit in data.index.to_series().unique():
            unit_data = data.loc[index_unit]
            unit_data_new = pd.DataFrame([])
            
            print('whole cycle of unit {} is {}, and whole length is {}'.format(index_unit,unit_data['cycle'].unique().max(),len(unit_data)))
            data_cycle_new = pd.DataFrame([])
            for cycle_index in unit_data['cycle'].unique():
                #print('this cycle: No.',cycle_index)             
                data_cycle = unit_data.loc[unit_data['cycle']==cycle_index].copy()
                data_cycle.drop(['cycle'],inplace=True,axis=1)            
                data_columns_new = pd.DataFrame([])
                              
                data_cycle['x_column'] = np.linspace(1,len(data_cycle),len(data_cycle)) #新建的位置都在最后一列  -1
                for x in range(data_cycle.shape[1]-1):  
                    '''
                    #-1对应的是x_columns 所以这里减去1本身,x_columns那一列不需要再sample
                    #因为一个cycle里的Fc flight class是一个值，所以不需要担心这个值会改变
                    #不能用cycle列直接作为x，因为这里的cycle是固定的一个值，所以此时需要构建新的x从1到len(data) 
                    #注意cycle不在第一（0）列，在中间的,所以需要一开始drop掉， 后面直接把cycle定义到第0列即可
                    '''
                    data_couple = (data_cycle.iloc[:,[-1,x]].values).tolist()   #跳过'cycle'列
                    sampled = LTTB(data_couple,len(data_couple)//self.down_sampling_rate)     
                    df_sampled = pd.DataFrame(np.array(sampled)[:,1]) #提取被down sample的数据
                    data_columns_new = pd.concat([data_columns_new,df_sampled],axis=1) 
 
                    #print('length of No. {} cycle from {} to {}'.format(cycle_index,len(data_cycle),len(df_sampled)))
                    #print('how many columns are sampled:', data_cycle_new.shape[1], 'still remain {} columns unsampled'.format(data_cycle.shape[1]-1-data_cycle_new.shape[1]))#'cycle'
                
                data_columns_new.insert(0,'cycle',np.ones(len(data_columns_new))*cycle_index, allow_duplicates=False)
                data_cycle_new = pd.concat([data_cycle_new,data_columns_new],axis=0)
             
            unit_data_new = data_cycle_new.copy()
            unit_data_new['unit'] = [index_unit]*len(unit_data_new) 
            data_new = pd.concat([data_new,unit_data_new],axis=0)
                          
        data_new.set_index(['unit'],inplace=True,drop=True) 
        columns_order = data.columns.to_list()
        columns_order.remove('cycle')  #注意 这里不能跟上一行写一起,否则返回None
        columns_order = ['cycle'] + columns_order  
        data_new.columns = columns_order 
        print('finally:')
        #print(data_new.columns)       
        print(data_new.shape)
        del data
        return data_new
    
    
    def save_data(self,df_dev,df_test):
        df_dev.reset_index(inplace=True,drop=False) #回收index列 unit列
        df_test.reset_index(inplace=True,drop=False)

        with open( "./DS02_{}_rate{}_dataset_info.txt".format(self.down_sampling,self.down_sampling_rate), "w+") as f1:
            f1.write("\ntrain shape after sampled: {}".format(df_dev.shape))
            f1.write("\ntest shape after sampled: {}".format(df_test.shape))
            f1.write("\ncolumn_name: {}".format(df_dev.columns))
            f1.write("\nunits in train_data: {}".format(df_dev.index.unique()))
            f1.write("\nunits in test_data: {}".format(df_test.index.unique()))

        print('dataset_info has been saved!') 
        
        # Create a new file hdf5 dataset
        f2 = h5py.File('Data/DS02/DS02_{}_rate{}.h5'.format(self.down_sampling,self.down_sampling_rate), 'w')
        f2.create_dataset('df_dev', data=df_dev)
        f2.create_dataset('df_test', data=df_test)
        data_columns=np.array(df_dev.columns.to_list())
        ds = f2.create_dataset('columns_name', data_columns.shape , dtype=h5py.special_dtype(vlen=str))
        ds[:] = data_columns
        f2.close()
        
        
    def process(self):
        if self.down_sampling=='simple_sampling':
            self.dow_sampling_rate=2
        
        df_dev, df_test = self.load_engine()
        
        if self.down_sampling=='LTTB_sampling':
            df_dev = self.LTTB_sampling(df_dev)
            df_test = self.LTTB_sampling(df_test)
        elif self.down_sampling=='simple_sampling':
            df_dev = self.SDS_sampling(df_dev) 
            df_test = self.SDS_sampling(df_test)
            
        #df_dev.reset_index(inplace=True,drop=False) #回收index列 unit列
        #df_test.reset_index(inplace=True,drop=False)
            
        print(df_dev.shape)
        print(df_test.shape)
        
        self.save_data(df_dev,df_test)
        #return df_dev, df_test
        
    ###保存h5文件  后续直接读取它即可

     
if __name__ == "__main__":
    filename = 'autodl-tmp/Data/DS02/N-CMAPSS_DS02-006.h5' #'Data/DS02/' + "N-CMAPSS_DS02-006.h5" 
    downsampling = DownSampling(
                    data_path=filename,
                    down_sampling='LTTB_sampling',#'LTTB_sampling','simple_sampling'
                    down_sampling_rate=6,  #6:1min;12:2min;30:5min;36:6min;60:10min;
                    )
                    
    #df_dev, df_test = downsampling.process()  
    downsampling.process()