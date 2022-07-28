# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.lstm = nn.LSTM(1, 32, batch_first=True,
                            bidirectional=False)
        self.linear = nn.Sequential(
            nn.Linear(32, 8),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        # 初始输入格式为(batch_size , length)
        # print(x.shape)

        x = x.view(x.shape[0], 60, -1)
        # print(x.shape)
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        # print(out.shape)
        out = self.linear(out)

        # multi-sample dropout 不一定有效
        # for i, dropout in enumerate(self.dropouts):
        #     if i == 0:
        #         res = dropout(out)
        #         res = self.linear(res)
        #     else:
        #         temp_out = dropout(out)
        #         res = res + self.linear(temp_out)
        # return res

        return out
