import argparse
import os
import torch
import pandas as pd
import numpy as np
from exp.exp_informer import Exp_Informer

### DARWIN AG DS02
'''2022.11.19
1.hs 0和1的交界点，往前往后4096个值，一共长度8192；
2.AG数据不够怎么解决？  9个unit，6train，3test；【40960】--->24*6个train
3.batch_size 24
4.enc_in和dec_in 对应d_features,ag还加了一列hi,所以是38；
5.attn=full

'''

parser = argparse.ArgumentParser(description='[Informer] data (time series) augmentation')
parser.add_argument('--model', type=str, required=False, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--HI_labeling_style', type=str, required=False, default='HI_pw_quadratic', help='HI style, options[HI_linear, HI_quadratic,HI_pw-linear,HI_pw_quadratic]')
parser.add_argument('--data_path', type=str, default='Data/DS02/')  #['autodl-tmp/Data/','Data/DS02/']
parser.add_argument('--root_path', type=str, default='Case/DARWIN/DARWIN_DS02/', help='root path of the data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='HI', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='autodl-tmp/checkpoints/', help='location of model checkpoints')

parser.add_argument('--down_sampling', type=str, default='LTTB_sampling', help='LTTB_sampling, simple_sampling')
parser.add_argument('--down_sampling_rate', type=int, default=6, help='whether to use down sampling to same memory, True 10s-->20s')
parser.add_argument('--stride', type=int, default=1, help='stride for time windows stride')
parser.add_argument('--seq_len', type=int, default=4096, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=4096, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length') #without mask
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--enc_in', type=int, default=38, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=38, help='decoder input size')
parser.add_argument('--c_out', type=int, default=38, help='output size') #14sensors+1HI
parser.add_argument('--validation_split', type=float, default=0.3) #0.3

parser.add_argument('--d_model', type=int, default=1024, help='dimension of model') #1024
parser.add_argument('--n_heads', type=int, default=8, help='num of heads') #8
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers') #2
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') #1
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn') #2048
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor') #5
parser.add_argument('--padding', type=int, default=0, help='padding type') #0  
parser.add_argument('--distil', type=str, default=True, help='whether to use distilling in encoder, using this argument means not using distilling')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout') #default=0.05
# parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--attn', type=str, default='full', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation') #gelu
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times') 
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=24, help='batch size of train input data') #default=32
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')#fault=0.0001
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function') #mse
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu') #default=True
parser.add_argument('--gpu', type=int, default=0, help='gpu') #default=0
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False) #default=False
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--dataset_name', type=str, default= 'DS02', help='target dataset')
parser.add_argument('--sensor_features', type=str, default= [''], help='sensor feature used')
parser.add_argument('--normal_style', type=str, default='StandardScaler',help='MinMaxScale or StandardScaler')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
# args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Informer     

for ii in range(args.itr):
    ## ii被迭代次数控制，itr default = 2   test_0,test_1
      
    ## 添加 learning rate,batch size
    name = '{}_{}_ft{}_sl{}_lr{}_bs{}_dm{}_at{}_fc{}_eb{}_dt{}_{}'.format(
                                                                    args.model, args.HI_labeling_style, args.features, 
                                                                    args.seq_len, args.learning_rate,args.batch_size,
                                                                    args.d_model, args.attn, args.factor, 
                                                                    args.embed, args.distil, args.des)
    
    setting = name + '_{}'.format(ii)


    exp = Exp(args) # set experiments
    print('\n')
    print('>>>>>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)            
    
    print('\n')
    print('>>>>>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
           
    torch.cuda.empty_cache()


all_df_metrics = pd.DataFrame()
for i in range(args.itr):  #args.itr
    file_name = name + '_{}'.format(i)
    metrics = np.load(args.root_path + 'results/{}/{}/metrics.npy'.format(args.dataset_name,file_name))
    df_metrics = pd.DataFrame(data = [metrics], columns = ['mae', 'mse', 'rmse', 'mape', 'mspe'])
    df_metrics['pred_len'] = args.pred_len
    
    all_df_metrics = all_df_metrics.append(df_metrics)
    
    #del the files
    for root, dirs, files in os.walk(args.root_path + 'results/{}/{}'.format(args.dataset_name,file_name), topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
    os.rmdir(args.root_path + 'results/{}/{}'.format(args.dataset_name,file_name))

all_df_metrics.to_csv(args.root_path + '/results_analyse/{}.csv'.format(name),index = False, header = True)