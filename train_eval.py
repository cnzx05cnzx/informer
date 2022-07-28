# coding: UTF-8
import numpy as np
import torch
import torch.nn.functional as F
import time


def train(config, model, train_iter, vaild_iter):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率

    best_loss = 1000
    best_num = []
    cnt = 0  # 记录多久没有模型效果提升
    stop_flag = False  # 早停标签

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        # 训练-------------------------------
        model.train()
        t_a = time.time()
        total_loss, total_acc = 0, 0

        for i, (data, pos) in enumerate(train_iter):
            outputs = model(data.to(config.device))
            pos = pos.to(config.device)
            # print(outputs)
            # print(pos)
            model.zero_grad()
            loss = F.mse_loss(outputs[0], pos)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        total_loss = total_loss / len(train_iter)

        t_b = time.time()
        msg = 'Train Loss: {0:>5.2}, Time: {1:>7.2}'
        print(msg.format(total_loss, t_b - t_a))

        # 验证 - ------------------------------
        model.eval()
        t_a = time.time()
        total_loss = 0
        temp_num = []
        for i, (data, pos) in enumerate(vaild_iter):
            with torch.no_grad():
                outputs = model(data.to(config.device))
                pos = pos.to(config.device)
                # print(outputs)
                # print(pos)
                loss = F.mse_loss(outputs[0], pos)
                # print(outputs.cpu.numpy())
                temp_num.append(outputs[0].cpu().numpy().tolist())

            total_loss += float(loss.item())

        total_loss = total_loss / len(vaild_iter)

        t_b = time.time()
        msg = 'Eval Loss: {0:>5.2},  Time: {1:>7.2}'
        print(msg.format(total_loss, t_b - t_a))

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), config.save_path)
            best_num = temp_num
            print('效果提升,保存最优参数')
            cnt = 0
        else:
            cnt += 1
            if cnt > config.early_stop:
                print('模型已无提升停止训练,验证集loss:%.2f' % best_loss)
                stop_flag = True
                break
        # model.load_state_dict(torch.load(config.save_path))
        # config.learning_rate = config.learning_rate * 0.9
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if stop_flag:
        pass
    else:
        print('训练结束,验证集loss:%.2f' % best_loss)
    return best_num


def test(config, model, test_iter):
    # 测试-------------------------------
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    t_a = time.time()
    total_acc, total_loss = 0, 0
    for i, (data, pos) in enumerate(test_iter):
        with torch.no_grad():
            outputs = model(data.to(config.device))
            pos = pos.to(config.device)
            loss = F.cross_entropy(outputs, pos)

        true = pos.data.cpu()
        predict = torch.max(outputs.data, 1)[1].cpu()
        total_loss += float(loss.item())
        total_acc += torch.eq(predict, true).sum().float().item()

    total_acc = total_acc / len(test_iter.dataset)
    total_loss = total_loss / len(test_iter)

    t_b = time.time()
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%},  Time: {2:>7.2}'
    print(msg.format(total_loss, total_acc, t_b - t_a))


#  投票选举或平均选举
def mul_test(config, models, name_lists, test_iter, type='vote'):
    # 测试-------------------------------
    cnts = len(models)
    for model, name in zip(models, name_lists):
        model.load_state_dict(torch.load('./saved_dict/{}.pkl'.format(name)))
        model.eval()

    t_a = time.time()
    total_acc, total_loss = 0, 0
    for step, (data, pos) in enumerate(test_iter):
        with torch.no_grad():

            res = torch.zeros(config.batch_size, config.num_classes)
            if type == 'vote':
                all_out = []
                all_loss = []
                for model in models:
                    outputs = model(data.to(config.device))
                    temp = torch.max(outputs.data, 1)[1].cpu()
                    all_out.append(temp)
                    all_loss.append(outputs)
                predict = []

                for i in range(config.batch_size):
                    temp = list(0 for i in range(config.batch_size))
                    for j in range(cnts):
                        temp[all_out[j][i]] += 1
                    tar = temp.index(max(temp))
                    predict.append(tar)
                    cnt = 0
                    for j in range(cnts):
                        if tar == all_out[j][i]:
                            res[i] += all_loss[j][i]
                            cnt += 1
                    res[i] = res[i] / cnt

                predict = torch.from_numpy(np.array(predict))
                pos = pos.to(config.device)
                loss = F.cross_entropy(res, pos)
            elif type == 'mean':
                for model in models:
                    outputs = model(data.to(config.device))
                    res = res + outputs
                res = res / cnts
                predict = torch.max(outputs.data, 1)[1].cpu()
                pos = pos.to(config.device)
                loss = F.cross_entropy(res, pos)

        true = pos.data.cpu()
        total_loss += float(loss.item())
        total_acc += torch.eq(predict, true).sum().float().item()

    total_acc = total_acc / len(test_iter.dataset)
    total_loss = total_loss / len(test_iter)

    t_b = time.time()
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%},  Time: {2:>7.2}'
    print(msg.format(total_loss, total_acc, t_b - t_a))
