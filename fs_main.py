'''Train Adversarially Robust Models with Feature Scattering'''
from __future__ import print_function
import time
import numpy as np
import random
from load_data import load_data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import copy
from torch.autograd import Variable
from PIL import Image

import os
import argparse
import datetime

from tqdm import tqdm
from models import *

import utils
from utils import softCrossEntropy
from utils import one_hot_tensor
from attack_methods import Attack_FeaScatter

torch.set_printoptions(threshold=10000)
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Feature Scatterring Training')

# add type keyword to registries
parser.register('type', 'bool', utils.str2bool)

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--adv_mode',
                    default='feature_scatter',
                    type=str,
                    help='adv_mode (feature_scatter)')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--data_dir', type=str, help='data path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass (-1: from scratch; K: checkpoint-K)')
parser.add_argument('--max_epoch',
                    default=200,
                    type=int,
                    help='max number of epochs')
parser.add_argument('--save_epochs', default=10, type=int, help='save period')
parser.add_argument('--decay_epoch1',
                    default=60,
                    type=int,
                    help='learning rate decay epoch one')
parser.add_argument('--decay_epoch2',
                    default=90,
                    type=int,
                    help='learning rate decay point two')
parser.add_argument('--decay_rate',
                    default=0.1,
                    type=float,
                    help='learning rate decay rate')
parser.add_argument('--batch_size_train',
                    default=128,
                    type=int,
                    help='batch size for training')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    help='momentum (1-tf.momentum)')
parser.add_argument('--weight_decay',
                    default=2e-4,
                    type=float,
                    help='weight decay')
parser.add_argument('--log_step', default=100, type=int, help='log_step')

# number of classes and image size will be updated below based on the dataset
parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--image_size', default=32, type=int, help='image size')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')  # concat cascade
parser.add_argument('--model', default='preactresnet18', type=str, choices=['preactresnet18', 'wrn28_10'],
                    help='dataset')  # concat cascade
parser.add_argument('--ls_factor', default=0.5, type=float, help='label smoothing factor')
parser.add_argument('--random_start', action='store_true', help='random start?')

args = parser.parse_args()
log = open(os.path.join(args.model_dir, 'log.txt'), 'w')

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


if args.dataset == 'cifar10':
    print_log('------------cifar10---------', log)
    args.num_classes = 10
    args.image_size = 32
elif args.dataset == 'cifar100':
    print_log('----------cifar100---------', log)
    args.num_classes = 100
    args.image_size = 32
if args.dataset == 'svhn':
    print_log('------------svhn10---------', log)
    args.num_classes = 10
    args.image_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

# Data
print_log('==> Preparing data..', log)
trainloader, _, _, _, num_classes = load_data(args.batch_size_train, 2, args.dataset, args.data_dir, valid_labels_per_class=0)
print_log('==> Building model..', log)

if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
    print_log(f"---{args.model}-----", log)
    if args.model.lower() == 'preactresnet18':
        basic_net = preactresnet18(num_classes=args.num_classes, per_img_std=True)
    elif args.model.lower() == 'wrn28_10':
        basic_net = wrn28_10(num_classes=args.num_classes, per_img_std=True)
    else:
        raise ValueError('model not supported')


def print_para(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)
        break


basic_net = basic_net.to(device)

# config for feature scatter
config_feature_scatter = {
    'train': True,
    'epsilon': 8.0,
    'num_steps': 1,
    'step_size': 8.0,
    'random_start': args.random_start,
    'ls_factor': args.ls_factor,
}

if args.adv_mode.lower() == 'feature_scatter':
    print_log('-----Feature Scatter mode -----', log)
    print_log(config_feature_scatter, log)
    net = Attack_FeaScatter(basic_net, config_feature_scatter)
else:
    print_log('-----OTHER_ALGO mode -----', log)
    raise NotImplementedError("Please implement this algorithm first!")

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

if args.resume and args.init_model_pass != '-1':
    # Load checkpoint.
    print_log('==> Resuming from checkpoint..', log)
    f_path_latest = os.path.join(args.model_dir, 'latest')
    f_path = os.path.join(args.model_dir,
                          ('checkpoint-%s' % args.init_model_pass))
    if not os.path.isdir(args.model_dir):
        print_log('train from scratch: no checkpoint directory or file found', log)
        exit(0)
    elif args.init_model_pass == 'latest' and os.path.isfile(f_path_latest):
        checkpoint = torch.load(f_path_latest)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        print_log('resuming from epoch %s in latest' % start_epoch, log)
    elif os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        print_log('resuming from epoch %s' % (start_epoch - 1), log)
    elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
        print_log('train from scratch: no checkpoint directory or file found', log)

soft_xent_loss = softCrossEntropy()


def train_fun(epoch, net):
    print_log('\nEpoch: %d' % epoch, log)
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # update learning rate
    if epoch < args.decay_epoch1:
        lr = args.lr
    elif epoch < args.decay_epoch2:
        lr = args.lr * args.decay_rate
    else:
        lr = args.lr * args.decay_rate * args.decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    def get_acc(outputs, targets):
        _, predicted = outputs.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        acc = 1.0 * correct / total
        return acc

    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        adv_acc = 0

        optimizer.zero_grad()

        # forward
        outputs, loss_fs = net(inputs.detach(), targets)

        optimizer.zero_grad()
        loss = loss_fs.mean()
        loss.backward()
        if torch.isnan(loss):
            print_log("\nnan loss!!!", log)
            exit(0)

        optimizer.step()

        train_loss = loss.item()
        iterator.set_description(f"{train_loss:.2f}")

        duration = time.time() - start_time
        if batch_idx % args.log_step == 0:
            if adv_acc == 0:
                adv_acc = get_acc(outputs, targets)

            nat_outputs, _ = net(inputs, targets, attack=False)
            nat_acc = get_acc(nat_outputs, targets)
            print_log(
                "epoch %d, step %d, lr %.4f, duration %.2f, training nat acc %.2f, training adv acc %.2f, training adv loss %.4f"
                % (epoch, batch_idx, lr, duration, 100 * nat_acc,
                   100 * adv_acc, train_loss), log)

    if epoch % args.save_epochs == 0 or epoch >= args.max_epoch - 2:
        print_log('Saving..', log)
        f_path = os.path.join(args.model_dir, ('checkpoint-%s' % epoch))
        state = {
            'net': net.state_dict(),
            # 'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, f_path)

    if epoch >= 0:
        print_log('Saving latest @ epoch %s..' % (epoch), log)
        f_path = os.path.join(args.model_dir, 'latest')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, f_path)


for epoch in range(start_epoch, args.max_epoch):
    train_fun(epoch, net)
