####单独保存时间窗  也是存起来给h5py  time_window 'train' 'test' 不考虑split vali
import numpy as np
import pandas as pd
import h5py

class WindowSliding():  
    def __init__(self,
                 down_sampling,
                 down_sampling_rate,
                 stride,
                 seq_len,
                 label_len,
                 pred_len,
                  ):
    
        self.down_sampling = down_sampling
        self.down_sampling_rate = down_sampling_rate
        self.stride=stride
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len


    def load_data(self):
        hdf = h5py.File('Data/DS02/DS02_{}_rate{}.h5'.format(self.down_sampling,self.down_sampling_rate), "r")
        sampled_train = np.array(hdf.get('df_dev'))
        sampled_test = np.array(hdf.get('df_test'))
        columns_name = np.array(hdf.get('columns_name'))
        columns_name = list(np.array(columns_name, dtype='U20'))
        hdf.close()

        sampled_train = pd.DataFrame(data=sampled_train,columns=columns_name)
        sampled_test = pd.DataFrame(data=sampled_test,columns=columns_name)
        
        sampled_train.set_index(['unit'],inplace=True,drop=True)
        sampled_test.set_index(['unit'],inplace=True,drop=True)
        
        print(sampled_train.shape)
        print(sampled_test.shape)
        return sampled_train, sampled_test
        

    def transform_data2window(self,data):
        ### enc, dec for save the precessed data(time window) 96*14 
        enc,dec = [],[]

        #Loop through each unit
        num_units = len(data.index.to_series().unique())
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
    
    
    def save_data(self,data_columns,train_enc,train_dec,test_enc,test_dec):       
        # Create a new file hdf5 dataset
        f2 = h5py.File('autodl-tmp/Data/DS02_{}_rate{}_windows_sl{}_ll{}_pl{}.h5'.format(self.down_sampling,self.down_sampling_rate,\
                                                                                   self.seq_len,self.label_len,self.pred_len), 'w')
        f2.create_dataset('train_enc', data=train_enc)
        f2.create_dataset('train_dec', data=train_dec)
        f2.create_dataset('test_enc', data=test_enc)
        f2.create_dataset('test_dec', data=test_dec)
        
        data_columns=np.array(data_columns)
        ds = f2.create_dataset('columns_name', data_columns.shape , dtype=h5py.special_dtype(vlen=str))
        ds[:] = data_columns
        f2.close()
        
    
    def process(self):
        df_dev, df_test = self.load_data()
        data_columns = df_dev.columns.to_list()
        train_enc,train_dec = self.transform_data2window(df_dev)
        test_enc,test_dec = self.transform_data2window(df_test)
        
        #print(len(train_enc),len(train_enc[0],len(train_enc[0][0])))
        print(len(train_dec))
                
        #print(len(test_enc),len(test_enc[0],len(test_enc[0][0])))
        print(len(test_dec))
        
        self.save_data(data_columns,train_enc,train_dec,test_enc,test_dec)


if __name__ == "__main__":
    window_sliding = WindowSliding(
                        down_sampling='LTTB_sampling',#'LTTB_sampling','simple_sampling'
                        down_sampling_rate=5,
                        stride = 5,
                        seq_len=256,
                        label_len=128,
                        pred_len=256,
                        )
    
    window_sliding.process()
    
    