import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from data_load import get_dataloader
import argparse

from importlib import import_module
from train_eval import train, test


def seed_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # benchmark加速 deterministic稳定 enabled非确定性
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='transformer', required=False, help='choose a models')

    args = parser.parse_args()
    choose = import_module('models.' + args.model)
    config = choose.Config()

    seed_init(config.seed)

    model = choose.Model(config).to(config.device)
    TrainL, ValidL, TestL, test_before, transform = get_dataloader(config)

    train(config, model, TrainL, ValidL)
    test_after = test(config, model, TestL)

    test_before = np.asarray(test_before).reshape(len(test_before), 1)
    test_after = np.asarray(test_after).reshape(len(test_after), 1)

    # print(test_before[:-20])
    # print(test_after[:-20])

    test_before = transform.inverse_transform(test_before)
    test_after = transform.inverse_transform(test_after)

    plt.figure()
    plt.title('LSTM Model')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Temperate', fontsize=18)
    plt.plot(test_before)
    plt.plot(test_after)

    plt.show()
