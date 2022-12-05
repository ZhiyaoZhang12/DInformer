# -*- coding: utf-8 -*-
###model
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from torch.autograd import Variable
import numpy as np
import random

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, is_perception,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.is_perception = is_perception
        
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )


        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc,  x_dec,  
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        

        # if self.output_attention:
        #     return dec_out[:,-self.pred_len:,:], attns
        # else:
        #     return dec_out[:,-self.pred_len:,:] # [B, L, D]
        
        if self.output_attention:
            if self.is_perception == False:
                return dec_out[:,-self.pred_len:,:], attns
            elif self.is_perception == True:
                 return dec_out[:,(-self.pred_len-self.label_len):,:], attns              
        else:
            if self.is_perception == False:
                return dec_out[:,-self.pred_len:,:] # [B, L, D]
            elif self.is_perception == True:
                return dec_out[:,(-self.pred_len-self.label_len):,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, is_perception,
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.is_perception = is_perception

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            if self.is_perception == False:
                return dec_out[:,-self.pred_len:,:], attns
            elif self._perception == True:
                 return dec_out[:,(-self.pred_len-self.label_len):,:], attns              
        else:
            if self.is_perception == False:
                return dec_out[:,-self.pred_len:,:] # [B, L, D]
            elif self._perception == True:
                return dec_out[:,(-self.pred_len-self.label_len):,:] # [B, L, D]

            

class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len, out_len):
        super(BiLSTM, self).__init__()
        self.pred_len = out_len
        self.seq_len = seq_len
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.linear2 = nn.Linear(self.seq_len,self.pred_len)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]  不对  output应该是[batch_size x pred_len x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        output = self.linear2(output.transpose(1,2))   #output.transpose(2,3)是 batch_size x output_size  x T   把T对应换为pred_len  
        output = output.transpose(1,2)
        return output
    

class DCNN2(torch.nn.Module):

    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 0
        features = 14
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out
    
class DCNN(nn.Module):
    def __init__(self, pred_len):
        super(DCNN, self).__init__()
        
        self.pred_len = pred_len 
        oc1,oc2,oc3,oc4 = 512,216,64,32
        
        self.conv1 = nn.Sequential(  # input shape (1, 48, 14) 
            nn.Conv2d(in_channels=1,out_channels=oc1,kernel_size=5,stride=1,padding=2,), # output shape (oc1, 48, 14) 
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),)    # 在 2x2 空间里向下采样, output shape (oc1, 24, 7)
        
        self.conv2 = nn.Sequential(  # input shape (oc1, 24, 7)
            nn.Conv2d(oc1, oc2, 5, 1, 2),  # output shape (oc2, 24, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),)  # output shape (oc2, 12, 3)
        
        
        self.conv3 = nn.Sequential(  # input shape (oc2, 12, 3)
            nn.Conv2d(oc2, oc3, 5, 1, 2),  # output shape (oc3, 12, 3)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),)  # output shape (oc3,6,1)
        
        self.conv4 = nn.Sequential(  # input shape (oc3, 6, 1)
            nn.Conv2d(oc3, oc4, 3, 1, 1),  # output shape (oc4, 6, 1)
            nn.ReLU(),)  # activation
            #nn.MaxPool2d(2),)  # output shape (oc3,6,1)   #不再下采样
        
        self.out = nn.Linear(oc4*6*1, self.pred_len)   # fully connected layer, output PDIs

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, oc4 * 6 * 1)
        output = self.out(x)   #(batch_size, pred_len)
        output=output.unsqueeze(2)  # #(batch_size, pred_len,1)
        return output

class DH_1(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, is_perception,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='full', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(DH_1, self).__init__()          
        #dec_out including the length of mask part, which is the perception part, then dec_out = label_len+pred_len;                                                   
        #out_len=pred_len;
        
        self.pred_len = out_len
        self.seq_len = seq_len

        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)  #valueEmbadding & positionalEmbadding (as transformer)
       
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.end_conv1 = nn.Conv1d(in_channels=seq_len, out_channels=out_len, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
        
    def forward(self, x_enc,  x_dec,  
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        enc_out = self.enc_embedding(x_enc)  #bs,seq_len,d_model   32,48,512
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.end_conv1(enc_out)   #bs,pred_len,d_model  32,12,512 
        dec_out = self.projection(dec_out) #bs,pred_len,c_out  32,12,1  

        
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out # [B, L, D]


class GRUConvCell(nn.Module):
    '''
    for TransGCU
    output: h
    '''

    def __init__(self, input_channel, output_channel):

        super(GRUConvCell, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size=3, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size=3, padding=1)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        self.activation = nn.Tanh()

    # function 1，2
    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = torch.sigmoid(rn)
        uns = torch.sigmoid(un)
        return rns, uns

    # function 3
    def output(self, x, h, r, u):

        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h = None):

        N, C, H, W = x.shape
        HC = self.output_channel  #64
                 
        if(h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)
   
        # function 4
        return u * h + (1 - u) * y


class GRUNet(nn.Module):
    '''
    for TransGCU
    output: o
    '''

    def __init__(self, hidden_size=64):

        super(GRUNet,self).__init__()

        self.gru_1 = GRUConvCell(input_channel=1,          output_channel=hidden_size)
        self.gru_2 = GRUConvCell(input_channel=hidden_size,output_channel=hidden_size)
        self.gru_3 = GRUConvCell(input_channel=hidden_size,output_channel=hidden_size)

        self.fc = nn.Conv2d(in_channels=hidden_size,out_channels=1,kernel_size=3,padding=1)

    def forward(self, x, h):

        if h is None:
            h = [None,None,None]
        
        # x [bs,1,seq_len,enc_in] 32,1,48,14
        h1 = self.gru_1( x,h[0])  #32,64,48,14
        h2 = self.gru_2(h1,h[1])  #32,64,48,14
        h3 = self.gru_3(h2,h[2])  #32,64,48,14

        o = self.fc(h3)   ##32,1,48,14

        return o,[h1,h2,h3]
        #return o

    

class TransGCU(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, is_perception,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='full', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(TransGCU, self).__init__()         
        
        
        self.pred_len = out_len
        self.seq_len = seq_len
        #self.label_len = label_len
        #self.is_perception = is_perception
        
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)  

       
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        

        self.gcu = GRUNet()
        self.end_conv = nn.Conv1d(in_channels=seq_len, out_channels=out_len, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
        
    def forward(self, x_enc,  x_dec,  
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
         
        enc_in = x_enc.unsqueeze(dim=1)    #[bs, channel,height,weight] 32,1,48,14
        enc_in,h_n = self.gcu(enc_in,h=None)  #  32,1,48,14 
        enc_in = enc_in.squeeze(dim=1) #32,48,14

        enc_out = self.enc_embedding(enc_in) #[bs,seq_len,d_model] 32,48,512
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) #[bs,seq_len,d_model] 32,48,512
        dec_out = self.end_conv(enc_out)   #bs,pred_len,d_model  32,12,512 
        dec_out = self.projection(dec_out) #bs,pred_len,c_out  32,12,1  

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out # [B, L, D]