import random

import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from model.lstm import Model
from train_eval import train
from data_load import get_dataloader, see_data


class Config(object):

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # self.device = torch.device('cpu')  # 设备
        self.filepath = './data/climate.csv'
        self.save_path = './save_dict/lstm.pkl'

        self.seed = 0
        self.dropout = 0.3  # 随机失活
        self.early_stop = 5  # 早停机制
        self.num_epochs = 20  # epoch数
        self.batch_size = 1  # batch大小
        self.learning_rate = 1e-3  # 学习率


def seed_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    config = Config()
    seed_init(config.seed)

    model = Model().to(config.device)
    TrainL, ValidL, test_before, transform = get_dataloader(config)
    test_after = train(config, model, TrainL, ValidL)

    # print(before[:-20])
    # print(after[:-20])

    # after = transform.inverse_transform(after)

    test_before = np.asarray(test_before).reshape(len(test_before), 1)
    test_before = transform.inverse_transform(test_before)
    test_after = transform.inverse_transform(test_after)

    plt.figure()
    plt.title('LSTM Model')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Temperate', fontsize=18)
    plt.plot(test_before)
    plt.plot(test_after)

    plt.show()
