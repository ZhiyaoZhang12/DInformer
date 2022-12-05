# -*- coding: utf-8 -*-
##DARWIN  AG DS02
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


class DataReader(Dataset): 
    def __init__(self, 
                 train_enc,
                 train_dec,
                 val_enc,
                 val_dec,
                 test_enc,
                 test_dec,
                 flag,      
                  ):

        self.train_enc = train_enc
        self.train_dec = train_dec
        self.val_enc   = val_enc
        self.val_dec   = val_dec
        self.test_enc  = test_enc
        self.test_dec  = test_dec
        
        self.flag = flag
        self.all_seq_x = []
        self._all_seq_y = []
        
        self.__read_data__() 
        
    
    def __read_data__(self):      
  
        if self.flag =='train':
            self.all_seq_x, self.all_seq_y = self.train_enc, self.train_dec
        elif self.flag =='val':
            self.all_seq_x, self.all_seq_y = self.val_enc, self.val_dec
        elif self.flag =='test':
            self.all_seq_x, self.all_seq_y = self.test_enc, self.test_dec    
                                                  
        
        print('falg:',self.flag, len(self.all_seq_x), len(self.all_seq_y) )

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
