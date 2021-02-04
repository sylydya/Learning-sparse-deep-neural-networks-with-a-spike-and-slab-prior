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


parser = argparse.ArgumentParser(description='Cifar ResNet Compression')

# Basic Setting
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--base_path', default='./result/cifar/', type = str, help = 'base path for saving result')
parser.add_argument('--model_path', default='test_run/', type = str, help = 'folder name for saving model')

# Resnet Architecture
parser.add_argument('-depth', default=20, type=int, help='Model depth.')

# Random Erasing
parser.add_argument('-p', default=0.5, type=float, help='Random Erasing probability')
parser.add_argument('-sh', default=0.4, type=float, help='max erasing area')
parser.add_argument('-r1', default=0.3, type=float, help='aspect of erasing area')

# Training Setting
parser.add_argument('--nepoch', default = 300, type = int, help = 'total number of training epochs')
parser.add_argument('--lr_decay_time', default = [150, 225], type = int, nargs= '+', help = 'when to multiply lr by 0.1')
parser.add_argument('--init_lr', default = 0.1, type = float, help = 'initial learning rate')

parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum in SGHMC')
parser.add_argument('--batch_train', default = 128, type = int, help = 'batch size for training')
parser.add_argument('--batch_test', default = 128, type = int, help = 'batch size for testing')
parser.add_argument('--temperature', default = 0.0001, type = float, help = 'temperature for SGHMC')

# Prior Setting
parser.add_argument('--sigma0', default = 0.0002, type = float, help = 'sigma_0 in prior')
parser.add_argument('--sigma1', default = 0.04, type = float, help = 'sigma_1 in prior')
parser.add_argument('--lambdan', default = 0.0001, type = float, help = 'lambda_n in prior')

# Proposal Setting
parser.add_argument('--Proposal_B', default = [400, 250], type = int, nargs= 2, help='proposal_b value at epoch 150 and 225')

args = parser.parse_args()

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class SGHMC(torch.optim.Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, temperature = 1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if temperature < 0.0:
            raise ValueError("Invalid temperature value: {}".format(temperature))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, temperature = temperature)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGHMC, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            temperature = group['temperature']
            lr = group['lr']

            alpha = 1 - momentum
            scale = np.sqrt(2.0*alpha*temperature/lr)

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                        buf.add_(torch.ones_like(buf).normal_().mul(scale))
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                else:
                    d_p = d_p.add(torch.ones_like(d_p).normal_().mul(scale))

                p.data.add_(-group['lr'], d_p)

        return loss



def model_eval(net, data_loader, device, loss_func):
    net.eval()
    correct = 0
    total_loss = 0
    total_count = 0
    for cnt, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        loss = loss_func(outputs, labels)
        prediction = outputs.data.max(1)[1]
        correct += prediction.eq(labels.data).sum().item()
        total_loss += loss.mul(images.shape[0]).item()
        total_count += images.shape[0]

    return  1.0 * correct / total_count, total_loss / total_count

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize,
                                          transforms.RandomErasing(probability=0.5, sh=0.4, r1=0.3)])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         normalize])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)


    np.random.seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_train, shuffle=True,                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_test, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_func = nn.CrossEntropyLoss().to(device)

    net = my_resnet.ResNet_sparse(args.depth, 10)
    net.to(device)

    lambda_n = args.lambdan
    prior_sigma = args.sigma1
    prior_sigma_0 = args.sigma0

    temperature = args.temperature

    step_lr = args.init_lr
    step_lr = step_lr/train_loader.dataset.data.shape[0]

    optimizer = SGHMC(net.parameters(), lr=step_lr, momentum=args.momentum, weight_decay=0, temperature = temperature)

    MH_loop = 1

    proposal_a = 0.1
    proposal_b = 2000
    epsilon = 1e-20


    current_ones = 0
    current_log_proposal = 0
    current_mask = {}
    new_mask = {}
    for name, para in net.named_parameters():
        probability = para.abs().mul(-proposal_b).exp().mul(proposal_a).add(1).pow(-1)
        current_ones = current_ones + para.numel()
        current_log_proposal = current_log_proposal + (probability+ epsilon).log().sum()
        new_mask[name] = torch.ones_like(para)
        current_mask[name] = torch.ones_like(para)


    PATH = args.base_path + args.model_path
    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    num_epochs = args.nepoch
    train_accuracy_path = np.zeros(num_epochs)
    train_loss_path = np.zeros(num_epochs)

    test_accuracy_path = np.zeros(num_epochs)
    test_loss_path = np.zeros(num_epochs)
    sparsity_path = np.zeros(num_epochs)

    torch.manual_seed(args.seed)

    NTrain = len(train_loader.dataset)
    best_accuracy = 0



    proposal_b_start = args.Proposal_B[0]
    proposal_b_end = args.Proposal_B[1]


    for epoch in range(num_epochs):
        net.train()
        epoch_training_loss = 0.0
        total_count = 0
        accuracy = 0


        if epoch <150:
            proposal_a = 0.1
            proposal_b = 2000
        if epoch >=150 and epoch < 225:
            proposal_a = 200
            proposal_b = proposal_b_start * (225.0 - epoch) / 75.0 + proposal_b_end * (1 - (225.0 - epoch) / 75.0)
            prior_sigma_0 = args.sigma0
        if epoch >= 225:
            proposal_b = proposal_b_end

        if epoch in args.lr_decay_time:
            for para in optimizer.param_groups:
                para['lr'] = para['lr'] / 10
        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)

            net.zero_grad()
            for MH_iter in range(MH_loop):

                output = net(input)
                loss = loss_func(output, target)
                prior = 0
                for name, para in net.named_parameters():
                    prior = prior + para.mul(current_mask[name]).pow(2).div(-2 * prior_sigma).sum() + para.mul(
                        1 - current_mask[name]).pow(2).div(-2 * prior_sigma_0).sum() \
                        + current_mask[name].sum().mul(-0.5*np.log(prior_sigma)) + (1-current_mask[name]).sum().mul(-0.5*np.log(prior_sigma_0))

                current_target = loss.mul(-NTrain).add(prior)
                current_loss = current_target.div(-MH_loop)

                new_ones = 0
                new_log_proposal = 0
                for name, para in net.named_parameters():
                    probability = para.abs().mul(-proposal_b).exp().mul(proposal_a).add(1).pow(-1)
                    new_mask[name] = torch.where(torch.rand_like(para) < probability, torch.ones_like(para),
                                                 torch.zeros_like(para))
                    new_ones = new_ones + new_mask[name].sum().item()
                    new_log_proposal = new_log_proposal + new_mask[name].add(probability - 1 + epsilon).abs().log().sum()
                net.update_mask(new_mask)

                new_output = net(input)
                new_loss = loss_func(new_output, target)
                new_prior = 0
                for name, para in net.named_parameters():
                    new_prior = new_prior + para.mul(new_mask[name]).pow(2).div(-2 * prior_sigma).sum() + para.mul(
                        1 - new_mask[name]).pow(2).div(-2 * prior_sigma_0).sum() \
                        + new_mask[name].sum().mul(-0.5*np.log(prior_sigma)) + (1-new_mask[name]).sum().mul(-0.5*np.log(prior_sigma_0))

                new_target = new_loss.mul(-NTrain).add(new_prior)
                new_loss = new_target.div(-MH_loop)

                with torch.no_grad():
                    log_MH_ratio = ((new_target - current_target) / temperature + (
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

            optimizer.step()

            epoch_training_loss += loss.mul(input.shape[0]).item()
            accuracy += output.data.argmax(1).eq(target.data).sum().item()
            total_count += input.shape[0]
            train_loss_path[epoch] = epoch_training_loss / total_count
            train_accuracy_path[epoch] = accuracy / total_count
        print("epoch: ", epoch, ", train loss: ", epoch_training_loss / total_count, "train accuracy: ",
              accuracy / total_count)

        with torch.no_grad():

            test_accuracy, test_loss = model_eval(net, test_loader, device, loss_func)
            test_loss_path[epoch] = test_loss
            test_accuracy_path[epoch] = test_accuracy
            print("epoch: ", epoch, ", test loss: ", test_loss, "test accuracy: ", test_accuracy)

            total_num_para = 0
            non_zero_element = 0
            for name, mask in net.named_masks():
                total_num_para += mask.numel()
                non_zero_element += mask.sum()
            print('sparsity:', non_zero_element.item() / total_num_para)
            sparsity_path[epoch] = non_zero_element.item() / total_num_para

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), PATH + 'best_model.pt')

            print('best accuracy:', best_accuracy)

        torch.save(net.state_dict(), PATH + 'model' + str(epoch) + '.pt')
        torch.save(current_mask, PATH + 'model' + str(epoch) + '_mask.pt')

    import pickle
    filename = PATH + 'result.txt'
    f = open(filename, 'wb')
    pickle.dump([train_loss_path, train_accuracy_path, test_loss_path, test_accuracy_path,sparsity_path], f)
    f.close()


if __name__ == '__main__':
    main()
