# -*- coding: utf-8 -*-
import argparse
import os
import torch
import pandas as pd
import numpy as np
from exp.exp_informer import Exp_Informer

#rate_data_iter = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
rate_data_iter = [1,0.5]

for i in range(len(rate_data_iter)):
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting') 
    parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD),dcnn,biLSTM,transgcu]')
    parser.add_argument('--HI_labeling_style', type=str, default='HI_pw_quadratic', help='HI style, options[HI_linear, HI_quadratic,HI_pw-linear,HI_pw_quadratic]')
    parser.add_argument('--data_path', type=str, default='Data/', help='data path of the data file')
    parser.add_argument('--root_path', type=str, default='Case/DARWIN/DARWIN_HI/', help='root path of the data file')
    parser.add_argument('--features', type=str, default='MS', \
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='HI', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='autodl-tmp/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=16, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=32, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=14, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=14, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor') #5  
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--dataset_name', type=str, default= 'FD001', help='target dataset')
    parser.add_argument('--sensor_features', type=str, default= ['sensor 2','sensor 3','sensor 4','sensor 7','sensor 8','sensor 9',
                                                                 'sensor 11','sensor 12','sensor 13','sensor 14','sensor 15','sensor 17',\
                                                                 'sensor 20','sensor 21'], help='sensor feature used')
    parser.add_argument('--data_augmentation', type=str, default=True, help='whether to use data augmentation')
    parser.add_argument('--is_padding', type=str, default=True, help='whether to padding degradation data')
    parser.add_argument('--is_descrsing', type=str, default=False,help='whether to make the sensor data in a uniform trend')
    parser.add_argument('--is_perception', type=str, default=False, help='whether to to save the perception results')
    parser.add_argument('--normal_style', type=str, default='StandardScaler',help='MinMaxScale or StandardScaler')
    parser.add_argument('--synthetic_data_path', default='Case/DARWIN/DARWIN_AG/results/FD001/Synthetic_HI_pw_quadratic_StandardScaler.csv') #(40,187)
    parser.add_argument('--rate_data', type=float, default=rate_data_iter[i],help='rate of data availability')  
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')                                                                     
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=8, help='train epochs') #8
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data') #default=32
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate') #0.0001
    parser.add_argument('--des', type=str, default='test',help='exp description')
    parser.add_argument('--loss', type=str, default='mse',help='loss function')  #default mse
    parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)    
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu') #default=True, False is CPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu') #default=0
    # parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False) #default=False
    parser.add_argument('--use_multi_gpu', help='use multiple gpus', default=False) #default=False
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7,8,9,10',help='device ids of multile gpus')
    
    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    
    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
    args.detail_freq = args.freq
    
    print('Args in experiment:')
    print(args)
    
    Exp = Exp_Informer     
    
    for ii in range(args.itr):     
        set_paras = '{}_rdata{}_sl{}ll{}pl{}bs{}at{}fc{}dt{}ispa{}ag{}pe{}nor{}'.format(args.model, args.rate_data,
            args.seq_len, args.label_len, args.pred_len,args.batch_size,
            args.attn, args.factor, 
            args.distil, args.is_padding,args.data_augmentation,args.is_perception,args.normal_style,args.des)
        
        setting = set_paras + '{}'.format(ii)
        
        print('Settings in this run:',setting)
    
    
        exp = Exp(args) # set experiments
        print('\n')
        print('>>>>>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        #exp.train(setting) 
        
        try:
            exp.train(setting)            
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        
        try:
            print('\n')
            print('>>>>>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting,flag='test_window')
            exp.test(setting,flag='test_whole')
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
               
        torch.cuda.empty_cache()
 

    def results_analysis(flag):
        all_df_metrics = pd.DataFrame()
        for k in range(args.itr):  #args.itr
            file_name = set_paras + '{}'.format(k)
        
            metrics = np.load(args.root_path+'results/{}/{}/metrics_{}.npy'.format(args.dataset_name,file_name,flag))
            df_metrics = pd.DataFrame(data = [metrics], columns = ['mae', 'mse', 'rmse', 'mape', 'mspe'])
            #df_metrics['pred_len'] = args.pred_len         
            
            all_df_metrics = all_df_metrics.append(df_metrics)
            
        means = pd.DataFrame([all_df_metrics.mean()],index=['mean'])  
        all_df_metrics = all_df_metrics.append(means)
        all_df_metrics['pred_len'] = [args.pred_len]*(len(all_df_metrics)-1) + ['mean']
        all_df_metrics.set_index('pred_len',inplace=True,drop=True)
        print('-------------{}----------------'.format(flag))
        print(all_df_metrics.iloc[-1,:])            
        return all_df_metrics
    
    test_window = results_analysis(flag='test_window')
    whole_test = results_analysis(flag='test_whole')
    test_results = pd.concat([test_window, whole_test],keys=['window','whole'],axis=1)
    test_results.to_csv(args.root_path+'/results_analyse/{}/{}.csv'.format(args.dataset_name,set_paras),index = True, header = True)
    
    if i == 0:
        Results_means = test_results.loc[test_results.index=='mean']
    else:
        Results_means = pd.concat([Results_means,test_results.loc[test_results.index=='mean']],axis=0)
    
Results_means.index = rate_data_iter 
Results_means.to_csv(args.root_path+'/results_analyse/{}/Means_results_{}.csv'.format(args.dataset_name,set_paras),index = True, header = True)


#del the files
for root, dirs, files in os.walk(args.root_path+'results/{}/'.format(args.dataset_name), topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:        
        os.rmdir(os.path.join(root, name))
