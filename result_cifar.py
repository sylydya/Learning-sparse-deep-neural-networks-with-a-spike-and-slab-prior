import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import torch.utils.data

#import torchvision.transforms as transforms
import transforms
import torchvision.datasets as datasets
import my_resnet
import os
import errno
from torch.utils.data.sampler import SubsetRandomSampler



a = 1

base_path = './result/cifar/'
model_path = 'test_run/'

PATH = base_path + model_path

import pickle

train_PATH = base_path + model_path
filename = train_PATH + 'result.txt'
f = open(filename, 'rb')
[train_loss_path, train_accuracy_path, test_loss_path, test_accuracy_path,
 sparsity_path] = pickle.load(f)
f.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 300
net = my_resnet.ResNet_sparse(20, 10)
net.to(device)
para_path = []
para_gamma_path = []
for para in net.parameters():
    para_path.append(np.zeros([num_epochs] + list(para.shape)))
    para_gamma_path.append(np.zeros([num_epochs] + list(para.shape)))

for epoch in range(num_epochs):
    if epoch % 20 == 0:
        print('load: ', epoch)
    net.load_state_dict(torch.load(PATH + 'model' + str(epoch) + '.pt'))
    mask = torch.load(PATH + 'model' + str(epoch) + '_mask.pt')

    for i, (name, para) in enumerate(net.named_parameters()):
        para_path[i][epoch,] = para.cpu().data.numpy()
        para_gamma_path[i][epoch,] = mask[name].cpu().data.numpy()

para_mean_gamma = []
total_num_para = 0
non_zero_element = 0
for temp in para_gamma_path:
    para_mean_gamma.append(temp[225:, ].mean(0) > 0.5)
    total_num_para += np.prod(para_mean_gamma[-1].shape)
    non_zero_element += para_mean_gamma[-1].sum()
print('sparsity:', non_zero_element.item() / total_num_para)


normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     normalize])

test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False, num_workers=4)
(images, labels) = test_loader.__iter__().next()
images, labels = images.to(device), labels.to(device)

starting_epoch = 225
num_epochs = 300
with torch.no_grad():
    net.eval()
    net.load_state_dict(torch.load(PATH + 'model' + str(starting_epoch) + '.pt'))
    mask = torch.load(PATH + 'model' + str(starting_epoch) + '_mask.pt')
    net.update_mask(mask)
    outputs = net(images)
    outputs = F.softmax(outputs)

    outputs_total = torch.zeros_like(outputs)
    outputs_total = outputs_total.add(outputs)
    for epoch in range(starting_epoch + 1, num_epochs):
        if epoch % 20 == 0:
            print('load: ', epoch)
        net.load_state_dict(torch.load(PATH + 'model' + str(epoch) + '.pt'))
        mask = torch.load(PATH + 'model' + str(epoch) + '_mask.pt')
        net.update_mask(mask)
        outputs = net(images)
        outputs = F.softmax(outputs)
        outputs_total = outputs_total.add(outputs)

    outputs_total = outputs_total.div(num_epochs - starting_epoch)
    prediction = outputs_total.data.max(1)[1]
    correct = prediction.eq(labels.data).sum().item()
    print(correct)


