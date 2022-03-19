import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

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

        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  
        output = self.linear(recurrent) 
        output2 = self.linear2(output.transpose(1,2))    
        output2 = output2.transpose(1,2)
        return output2

    
class DCNN(nn.Module):
    def __init__(self, pred_len):
        super(DCNN, self).__init__()
        
        self.pred_len = pred_len 
        oc1,oc2,oc3,oc4 = 512,216,64,32
        
        self.conv1 = nn.Sequential(  
            nn.Conv2d(in_channels=1,out_channels=oc1,kernel_size=5,stride=1,padding=2,), 
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2),)   
        
        self.conv2 = nn.Sequential(  
            nn.Conv2d(oc1, oc2, 5, 1, 2),  
            nn.ReLU(),  
            nn.MaxPool2d(2),)  
        
        
        self.conv3 = nn.Sequential(  
            nn.Conv2d(oc2, oc3, 5, 1, 2),  
            nn.ReLU(),  
            nn.MaxPool2d(2),)  
        
        self.conv4 = nn.Sequential(  
            nn.Conv2d(oc3, oc4, 3, 1, 1),  
            nn.ReLU(),)  
        
        self.out = nn.Linear(oc4*6*1, self.pred_len)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)  
        output = self.out(x)   
        output=output.unsqueeze(2)  
        return output


