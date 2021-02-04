import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import argparse
import errno

import os
from mask_module import Mask_Linear, Mask_Conv2d, Mask_BatchNorm2d, Mask_ReLU, Mask_AvgPool2d, Mask_Sequential


from collections import OrderedDict, namedtuple


parser = argparse.ArgumentParser(description='Simulation Regression')

# Basic Setting
parser.add_argument('--data_index', default=1, type = int, help = 'set data index')
args = parser.parse_args()

class my_Net(torch.nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self.fc1 = Mask_Linear(1000, 5, bias=False)
        self.fc2 = Mask_Linear(5,3, bias=False)
        self.fc3 = Mask_Linear(3,1, bias=False)
        self._masks = OrderedDict()

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def named_masks(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._masks.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def to(self, *args, **kwargs):
        super(my_Net, self).to(*args, **kwargs)
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)
        for name, para in self.named_masks():
            para.data = para.to(device)


    def update_mask(self, user_mask):
        for name, para in self.named_masks():
            para.data = user_mask[name]

    def get_mask(self):
        total_mask = {}
        for name, para in self.named_masks():
            total_mask[name] = para.clone()
        return total_mask



def main():
    data_index = args.data_index
    subn = 200


    NTrain = 10000
    Nval = 1000
    NTest = 1000
    TotalP = 1000

    x_train = np.matrix(np.zeros([NTrain, TotalP]))
    y_train = np.matrix(np.zeros([NTrain, 1]))

    x_val = np.matrix(np.zeros([Nval, TotalP]))
    y_val = np.matrix(np.zeros([Nval, 1]))

    x_test = np.matrix(np.zeros([NTest, TotalP]))
    y_test = np.matrix(np.zeros([NTest, 1]))

    temp = np.matrix(pd.read_csv("./data/structure/" + str(data_index) + "/x_train.csv"))
    x_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(data_index) + "/y_train.csv"))
    y_train[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(data_index) + "/x_val.csv"))
    x_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(data_index) + "/y_val.csv"))
    y_val[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(data_index) + "/x_test.csv"))
    x_test[:, :] = temp[:, 1:]
    temp = np.matrix(pd.read_csv("./data/structure/" + str(data_index) + "/y_test.csv"))
    y_test[:, :] = temp[:, 1:]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)




    np.random.seed(data_index)
    torch.manual_seed(data_index)

    net = my_Net()
    net.to(device)
    loss_func = nn.MSELoss()

    step_lr = 0.005

    step_lr = step_lr/NTrain
    optimization = torch.optim.SGD(net.parameters(), lr=step_lr)

    sigma = torch.FloatTensor([1]).to(device)
    # sigma.requires_grad = True

    max_loop = 100001
    PATH = './result/structure/'

    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    show_information = 100

    para_path = []
    para_gamma_path = []
    for para in net.parameters():
        para_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))
        para_gamma_path.append(np.zeros([max_loop // show_information + 1] + list(para.shape)))

    train_loss_path = np.zeros([max_loop // show_information + 1])
    val_loss_path = np.zeros([max_loop // show_information + 1])
    test_loss_path = np.zeros([max_loop // show_information + 1])


    MH_loop = 5
    lambda_n = 0.001
    prior_sigma = 0.01
    prior_sigma_0 = 0.001

    proposal_a = 1000
    proposal_b = 200

    total_ones = 0

    current_ones = 0
    current_log_proposal = 0
    current_mask = {}
    new_mask = {}
    name_list = []

    ones_list = []
    zeros_list = []

    for name, para in net.named_parameters():
        probability = para.abs().mul(-proposal_b).exp().mul(proposal_a).add(1).pow(-1)
        current_ones = current_ones + para.numel()
        total_ones = total_ones + para.numel()
        current_log_proposal = current_log_proposal + probability.log().sum()
        new_mask[name] = torch.ones_like(para)
        current_mask[name] = torch.ones_like(para)
        name_list.append(name)
        ones_list.append(para.numel())
        zeros_list.append(0)

    ones_list = np.array(ones_list)
    zeros_list = np.array(zeros_list)


    temperature = 0.01


    for iter in range(max_loop):
        if subn == NTrain:
            subsample = range(NTrain)
        else:
            subsample = np.random.choice(range(NTrain), size=subn, replace=False)

        net.zero_grad()


        if iter < 5000:
            proposal_b = 800
        if iter >=5000 and iter < 10000:
            proposal_b = 400
            prior_sigma_0 = 0.001
        if iter >=10000 and iter < 15000:
            proposal_b = 100
        if iter >= 15000 and iter < 20000:
            proposal_b = 50
        if iter >= 20000:
            proposal_b = 10


        for MH_iter in range(MH_loop):

            if iter < 30000:
                output = net(x_train[subsample,])
                loss = loss_func(output, y_train[subsample,])
                loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))
                prior = 0
                for name, para in net.named_parameters():
                    prior = prior + para.mul(current_mask[name]).pow(2).div(-2 * prior_sigma).sum() + para.mul(
                        1 - current_mask[name]).pow(2).div(-2 * prior_sigma_0).sum() \
                            + current_mask[name].sum().mul(-0.5 * np.log(prior_sigma)) + (
                                        1 - current_mask[name]).sum().mul(-0.5 * np.log(prior_sigma_0))
                current_target = loss.mul(-NTrain).add(prior)
                current_loss = current_target.div(-MH_loop)

                new_ones = 0
                new_log_proposal = 0
                for name, para in net.named_parameters():
                    probability = para.abs().mul(-proposal_b).exp().mul(proposal_a).add(1).pow(-1)
                    new_mask[name] = torch.where(torch.rand_like(para) < probability, torch.ones_like(para),
                                                 torch.zeros_like(para))
                    new_ones = new_ones + new_mask[name].sum().item()
                    new_log_proposal = new_log_proposal + new_mask[name].add(probability - 1).abs().log().sum()
                net.update_mask(new_mask)

                new_output = net(x_train[subsample,])
                new_loss = loss_func(new_output, y_train[subsample,])
                new_loss = new_loss.div(2 * sigma).add(sigma.log().mul(0.5))
                new_prior = 0
                for name, para in net.named_parameters():
                    new_prior = new_prior + para.mul(new_mask[name]).pow(2).div(-2 * prior_sigma).sum() + para.mul(
                        1 - new_mask[name]).pow(2).div(-2 * prior_sigma_0).sum() \
                                + new_mask[name].sum().mul(-0.5 * np.log(prior_sigma)) + (1 - new_mask[name]).sum().mul(
                        -0.5 * np.log(prior_sigma_0))
                new__target = new_loss.mul(-NTrain).add(new_prior)
                new_loss = new__target.div(-MH_loop)

                with torch.no_grad():
                    log_MH_ratio = ((new__target - current_target) / temperature + (
                                new_ones - current_ones) / temperature * np.log(lambda_n / (1 - lambda_n)) + (
                                                current_log_proposal - new_log_proposal)).item()

                test_stat = np.random.uniform(0, 1, 1)
                if np.log(test_stat) < log_MH_ratio:
                    new_loss.backward()
                    with torch.no_grad():
                        current_log_proposal = new_log_proposal
                        current_ones = new_ones
                        temp = current_mask
                        current_mask = new_mask
                        new_mask = temp
                else:
                    current_loss.backward()
                    net.update_mask(current_mask)

            if iter >= 30000:

                for temp_index, name in enumerate(name_list):
                    ones_list[temp_index] = current_mask[name].sum().item()
                    zeros_list[temp_index] = (current_mask[name].numel() - current_mask[name].sum()).item()
                zeros_cum_list = zeros_list.cumsum()
                ones_cum_list = ones_list.cumsum()

                output = net(x_train[subsample,])
                loss = loss_func(output, y_train[subsample,])
                loss = loss.div(2 * sigma).add(sigma.log().mul(0.5))
                prior = 0
                for name, para in net.named_parameters():
                    prior = prior + para.mul(current_mask[name]).pow(2).div(-2 * prior_sigma).sum() + para.mul(
                        1 - current_mask[name]).pow(2).div(-2 * prior_sigma_0).sum() \
                            + current_mask[name].sum().mul(-0.5 * np.log(prior_sigma)) + (
                                        1 - current_mask[name]).sum().mul(-0.5 * np.log(prior_sigma_0))
                current_target = loss.mul(-NTrain).add(prior)
                current_loss = current_target.div(-MH_loop)

                for name, para in net.named_parameters():
                    new_mask[name].data = current_mask[name].clone()

                move_type_stat = np.random.uniform(0, 1, 1)
                if current_ones == total_ones:
                    move_type = -1
                elif current_ones == 0:
                    move_type = 1
                else:
                    move_type = -1 * (move_type_stat < 1.0 / 3.0) + 0 * (
                                move_type_stat >= 1.0 / 3.0 and move_type_stat < 2.0 / 3.0) + 1 * (
                                            move_type_stat >= 2.0 / 3.0)

                if move_type == 1:
                    index_stat = (total_ones - current_ones) * np.random.uniform(0, 1, 1)
                    para_index = np.sum(zeros_cum_list < index_stat)
                    if para_index == 0:
                        position_index = np.int(np.floor(index_stat))
                    else:
                        position_index = np.int(np.floor(index_stat - zeros_cum_list[para_index - 1]))
                    position = tuple((current_mask[name_list[para_index]] < 0.5).nonzero()[position_index])
                    new_mask[name_list[para_index]][position] = 1.0
                    new_ones = current_ones + 1

                    new_log_proposal = np.log(1.0 / (total_ones - current_ones))
                    current_log_proposal = np.log(1.0 / (current_ones + 1))
                if move_type == -1:
                    index_stat = (current_ones) * np.random.uniform(0, 1, 1)
                    para_index = np.sum(ones_cum_list < index_stat)
                    if para_index == 0:
                        position_index = np.int(np.floor(index_stat))
                    else:
                        position_index = np.int(np.floor(index_stat - ones_cum_list[para_index - 1]))
                    position = tuple((current_mask[name_list[para_index]] > 0.5).nonzero()[position_index])
                    new_mask[name_list[para_index]][position] = 0.0
                    new_ones = current_ones - 1

                    new_log_proposal = np.log(1 / (current_ones))
                    current_log_proposal = np.log(1 / (total_ones - current_ones + 1))
                if move_type == 0:
                    index_stat_1 = (total_ones - current_ones) * np.random.uniform(0, 1, 1)
                    para_index_1 = np.sum(zeros_cum_list < index_stat_1)
                    if para_index_1 == 0:
                        position_index_1 = np.int(np.floor(index_stat_1))
                    else:
                        position_index_1 = np.int(np.floor(index_stat_1 - zeros_cum_list[para_index_1 - 1]))
                    position_1 = tuple((current_mask[name_list[para_index_1]] < 0.5).nonzero()[position_index_1])

                    index_stat_2 = (current_ones) * np.random.uniform(0, 1, 1)
                    para_index_2 = np.sum(ones_cum_list < index_stat_2)
                    if para_index_2 == 0:
                        position_index_2 = np.int(np.floor(index_stat_2))
                    else:
                        position_index_2 = np.int(np.floor(index_stat_2 - ones_cum_list[para_index_2 - 1]))
                    position_2 = tuple((current_mask[name_list[para_index_2]] > 0.5).nonzero()[position_index_2])

                    new_mask[name_list[para_index_1]][position_1] = 1.0
                    new_mask[name_list[para_index_2]][position_2] = 0.0

                    new_ones = current_ones
                    new_log_proposal = 0
                    current_log_proposal = 0

                net.update_mask(new_mask)
                new_output = net(x_train[subsample,])
                new_loss = loss_func(new_output, y_train[subsample,])
                new_loss = new_loss.div(2 * sigma).add(sigma.log().mul(0.5))
                new_prior = 0
                for name, para in net.named_parameters():

                    new_prior = new_prior + para.mul(new_mask[name]).pow(2).div(-2 * prior_sigma).sum() + para.mul(
                        1 - new_mask[name]).pow(2).div(-2 * prior_sigma_0).sum() \
                                + new_mask[name].sum().mul(-0.5 * np.log(prior_sigma)) + (1 - new_mask[name]).sum().mul(
                        -0.5 * np.log(prior_sigma_0))
                new__target = new_loss.mul(-NTrain).add(new_prior)
                new_loss = new__target.div(-MH_loop)

                with torch.no_grad():
                    log_MH_ratio = ((new__target - current_target) / temperature + (
                                new_ones - current_ones) / temperature * np.log(lambda_n / (1 - lambda_n)) + (
                                                current_log_proposal - new_log_proposal)).item()

                test_stat = np.random.uniform(0, 1, 1)
                if np.log(test_stat) < log_MH_ratio:
                    new_loss.backward()
                    with torch.no_grad():
                        current_ones = new_ones
                        temp = current_mask
                        current_mask = new_mask
                        new_mask = temp
                else:
                    current_loss.backward()
                    net.update_mask(current_mask)


        optimization.step()
        with torch.no_grad():
            for para in net.parameters():
                para.data = para + torch.FloatTensor(para.shape).to(device).normal_().mul(np.sqrt(step_lr * temperature))

        if iter % show_information == 0:
            print('iteration:', iter)
            with torch.no_grad():
                output = net(x_train)
                loss = loss_func(output, y_train)
                print("train loss:", loss)
                train_loss_path[iter // show_information] = loss.cpu().data.numpy()
                output = net(x_val)
                loss = loss_func(output, y_val)
                print("val loss:", loss)
                val_loss_path[iter // show_information] = loss.cpu().data.numpy()
                output = net(x_test)
                loss = loss_func(output, y_test)
                print("test loss:", loss)
                test_loss_path[iter // show_information] = loss.cpu().data.numpy()
                print('sigma:', sigma)

                for i, para in enumerate(net.parameters()):
                    para_path[i][iter // show_information,] = para.cpu().data.numpy()

                for i, (name,mask) in enumerate(net.named_masks()):
                    para_gamma_path[i][iter // show_information,] = mask.cpu().data.numpy()

                print('number of 1:', np.sum((np.max(para_gamma_path[0][0:(iter // show_information + 1),].mean(0) > 0.5, 0) > 0)))
                print('number of true:',
                      np.sum((np.max(para_gamma_path[0][0:(iter // show_information + 1),].mean(0) > 0.5, 0) > 0)[0:5]))



    import pickle

    filename = PATH + 'data_' + str(data_index) + "result.txt"
    f = open(filename, 'wb')
    pickle.dump([para_path, para_gamma_path, train_loss_path, val_loss_path, test_loss_path], f)
    f.close()


if __name__ == '__main__':
    main()