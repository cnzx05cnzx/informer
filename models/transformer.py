import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'transformer'
        self.filepath = './data/climate.csv'  # 总集

        self.save_path = './save_dict/' + self.model_name + '.pkl'  # 模型训练结果

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # self.device = torch.device('cpu')  # 设备

        self.seed = 721
        self.dropout = 0.5  # 随机失活
        self.early_stop = 5  # 早停机制
        self.num_classes = 1  # 类别
        self.num_epochs = 40  # epoch数
        self.batch_size = 32  # batch大小
        self.max_len = 24  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率

        self.embed_size = 1

        self.dim_model = 1
        self.hidden = 16
        self.num_head = 1
        self.num_encoder = 1


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Model(nn.Module):
    def __init__(self, config, feature_size=16, num_layers=1, dropout=0.1):
        super(Model, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.view(src.shape[0], src.shape[1], 1)
        src = torch.transpose(src, 0, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        print(src.shape)
        src = self.pos_encoder(src)
        print(src.shape)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        print(output.shape)
        output = self.decoder(output)
        print(output.shape)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
