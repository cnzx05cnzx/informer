from collections import Counter

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pandas as pd


class LSTMDataSet(Dataset):
    def __init__(self, data, data_targets):
        self.content = torch.FloatTensor(data)
        self.pos = torch.FloatTensor(data_targets)

    def __getitem__(self, index):
        return self.content[index], self.pos[index]

    def __len__(self):
        return len(self.pos)


def make_data(path):
    df = pd.read_csv(path)
    # 将DATA转为时间标签并设置其为索引列
    # df["Date Time"] = pd.to_datetime(df["Date Time"], dayfirst=True)
    # df.set_index("Date Time", inplace=True)

    # 按天取数据，用下一条数据填充缺失值
    # df = df.asfreq("d")
    df = df.fillna(method="bfill")
    return df


def see_data(path):
    df = pd.read_csv(path)
    # 将DATA转为时间标签并设置其为索引列
    print(df.head())


def get_dataloader(config):
    # see_data(filepath)
    data = make_data(config.filepath)[:5000]

    data = data[['T (degC)']]
    # print(data.head())

    dataset = data.values
    start_len = int(np.ceil(len(dataset) * .8))
    end_len = int(np.ceil(len(dataset) * .9))

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_train = dataset[:start_len, ]
    scaled_eval = dataset[start_len:end_len, ]
    scaled_test = dataset[end_len:, ]

    scaled_train = scaler.fit_transform(scaled_train)
    scaled_eval = scaler.transform(scaled_eval)
    scaled_test = scaler.transform(scaled_test)

    def solve_data(s_data):
        x_data = []
        y_data = []
        seq_len = 24
        for i in range(seq_len, len(s_data)):
            x_data.append(list(s_data[i - seq_len:i, 0]))
            y_data.append(s_data[i, 0])

        return x_data, y_data

    x, y = solve_data(scaled_train)
    train_set = LSTMDataSet(x, y)
    TrainLoader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False)

    x, y = solve_data(scaled_eval)
    eval_set = LSTMDataSet(x, y)
    EvalLoader = DataLoader(eval_set, batch_size=config.batch_size, shuffle=False)

    x, y = solve_data(scaled_test)
    test_set = LSTMDataSet(x, y)
    TestLoader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    # 针对只有整个数据集的处理

    print('Training {} samples...'.format(len(TrainLoader.dataset)))
    print('Evaling {} samples...'.format(len(EvalLoader.dataset)))
    print('Testing {} samples...'.format(len(TestLoader.dataset)))
    return TrainLoader, EvalLoader, TestLoader, y, scaler


if __name__ == "__main__":
    pass
