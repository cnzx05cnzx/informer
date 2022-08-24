# coding: UTF-8
import torch
import torch.nn as nn


class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # self.device = torch.device('cpu')  # 设备
        self.filepath = './data/climate.csv'
        self.save_path = './save_dict/lstm.pkl'

        self.seed = 0
        self.dropout = 0.5  # 随机失活
        self.early_stop = 5  # 早停机制
        self.num_epochs = 20  # epoch数
        self.batch_size = 32  # batch大小
        self.learning_rate = 1e-3  # 学习率


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(1, 32, batch_first=True,
                            bidirectional=False, num_layers=2)
        self.linear = nn.Sequential(
            nn.Linear(32, 8),
            nn.Dropout(config.dropout),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

        self.dropouts = nn.ModuleList([nn.Dropout(config.dropout) for _ in range(3)])

    def forward(self, x):
        # 初始输入格式为(batch_size , length)
        # print(x.shape)

        x = x.view(x.shape[0], 24, -1)
        # print(x.shape)
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]

        # 无 multi-sample dropout
        # out = self.linear(out)
        # out = out.view(out.shape[0])
        # return out

        # multi-sample dropout 不一定有效
        res = 0
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                res = dropout(out)
                res = self.linear(res)
            else:
                temp_out = dropout(out)
                res = res + self.linear(temp_out)

        res = res.view(res.shape[0])
        print(res.shape)
        return res


