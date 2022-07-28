from collections import Counter

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, Dataset
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
    df["Date"] = pd.to_datetime(df.Date, dayfirst=True)
    df.set_index("Date", inplace=True)

    # 按天取数据，用下一条数据填充缺失值
    df = df.asfreq("d")
    df = df.fillna(method="bfill")
    return df


def get_dataloader(config):
    # see_data(filepath)
    data = make_data(config.filepath)

    data = data[['Open']]

    dataset = data.values

    train_len = int(np.ceil(len(dataset) * .7))
    test_len = int(np.ceil(len(dataset) * .9))

    # train_data = dataset[:train_len, :]
    # test_data = dataset[train_len:test_len, :]

    scaler = MinMaxScaler(feature_range=(0, 1))

    # scaled_train = scaler.fit_transform(train_data)
    # scaled_test = scaler.transform(test_data)

    dataset = scaler.fit_transform(dataset)
    scaled_train = dataset[:train_len, :]
    scaled_test = dataset[train_len:test_len, :]

    def solve_data(data):
        x_data = []
        y_data = []
        seq_len = 60
        for i in range(seq_len, len(data)):
            x_data.append(list(data[i - seq_len:i, 0]))
            y_data.append(data[i, 0])

        return x_data, y_data

    a, b = solve_data(scaled_train)
    train_set = LSTMDataSet(a, b)
    TrainLoader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False)

    a, b = solve_data(scaled_test)
    test_set = LSTMDataSet(a, b)
    TestLoader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    # 针对只有整个数据集的处理

    print('Training {} samples...'.format(len(TrainLoader.dataset)))
    # print('Validing {} samples...'.format(len(ValidLoader.dataset)))
    print('Testing {} samples...'.format(len(TestLoader.dataset)))
    return TrainLoader, TestLoader, b, scaler


if __name__ == "__main__":
    pass
