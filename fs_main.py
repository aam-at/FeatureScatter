'''Train Adversarially Robust Models with Feature Scattering'''
from __future__ import print_function
import time
import numpy as np
import random
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
parser.add_argument('--data_dir', type=str, help='path where to store the data')
parser.add_argument('--root_dir', type=str, help='folder to store results')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass (-1: from scratch; K: checkpoint-K)')
parser.add_argument('--max_epoch',
                    default=200,
                    type=int,
                    help='max number of epochs')
parser.add_argument('--save_epochs', default=100, type=int, help='save period')
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
parser.add_argument('--log_step', default=10, type=int, help='log_step')

parser.add_argument('--model', type=str, default='preactresnet18', help='model name', choices=['wrn28_10', 'preactresnet18'])
parser.add_argument('--train',type=str, default='standard', choices=['standard', 'lamb', 'mixup', 'mixup_hidden'])
parser.add_argument('--mixup_alpha', type=float, default=0.0, help='alpha parameter for mixup')
parser.add_argument('--ls-factor', type=float, default=0.0, help='label smoothing factor')
parser.add_argument('--random-start', action='store_true', default=False, help='random start for the attack')

# number of classes and image size will be updated below based on the dataset
parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--image_size', default=32, type=int, help='image size')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')  # concat cascade
parser.add_argument('--job_id', type=str, default='')

args = parser.parse_args()


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

exp_name = args.dataset
exp_name += "_arch_" + str(args.model)
exp_name += '_train_'+str(args.train)
if args.train in ['mixup', 'mixup_hidden']:
    exp_name += '_a_'+ str(args.mixup_alpha)
exp_name += '_adv_mode_'+str(args.adv_mode)
if args.random_start:
    exp_name += '_rs'
exp_name += '_ls_'+str(args.ls_factor)
if args.job_id != None:
    exp_name += '_job_id_'+str(args.job_id)
args.model_dir = os.path.join(args.root_dir, exp_name)
if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
log = open(os.path.join(args.model_dir, 'log.txt'), 'w')
print(exp_name)

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

if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
elif args.dataset == 'svhn':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir,
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
elif args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir,
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir,
                                            train=False,
                                            download=True,
                                            transform=transform_test)

elif args.dataset == 'svhn':
    trainset = torchvision.datasets.SVHN(root=args.data_dir,
                                         split='train',
                                         download=True,
                                         transform=transform_train)
    testset = torchvision.datasets.SVHN(root=args.data_dir,
                                        split='test',
                                        download=True,
                                        transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size_train,
                                          shuffle=True,
                                          num_workers=2)

print_log('==> Building model..', log)

if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
    print_log(args.model, log)
    if args.model == 'wrn28_10':
        net = wrn28_10(num_classes=args.num_classes, widen_factor=10, per_img_std=True)
    elif args.model == 'preactresnet18':
        net = preactresnet18(num_classes=args.num_classes, per_img_std=True)


net = net.to(device)

# config for feature scatter
config_feature_scatter = {
    'epsilon': 8.0,
    'num_steps': 1,
    'step_size': 8.0,
    'random_start': args.random_start,
    'clip_min': 0.0,
    'clip_max': 255.0
}

if args.adv_mode.lower() == 'feature_scatter':
    print_log('-----Feature Scatter mode -----', log)
    attack = Attack_FeaScatter(net, config_feature_scatter)
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

    mixup = False
    mixup_hidden = False
    if args.train == 'mixup':
        mixup = True
    elif args.train == 'mixup_hidden':
        mixup_hidden = True

    iterator = tqdm(trainloader, ncols=0, leave=False)
    loss_ce = softCrossEntropy()
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        adv_acc = 0

        optimizer.zero_grad()

        # attack
        inputs_adv = attack(inputs, targets)
        # forward
        outputs, _, targets_reweighted = net(inputs, targets, mixup=mixup, mixup_hidden=mixup_hidden, mixup_alpha=args.mixup_alpha)
        if args.ls_factor > 0.0:
            targets_reweighted = utils.label_smoothing(targets_reweighted, targets_reweighted.size(1), args.ls_factor)
        outputs_adv, _, targets_reweighted2 = net(inputs_adv, targets, mixup=mixup, mixup_hidden=mixup_hidden, mixup_alpha=args.mixup_alpha)
        if args.ls_factor > 0.0:
            targets_reweighted2 = utils.label_smoothing(targets_reweighted2, targets_reweighted2.size(1), args.ls_factor)

        if args.train == 'standard':
            adv_loss = loss_ce(outputs_adv, targets_reweighted2.detach())
            loss = adv_loss
        elif args.train in ['lamb', 'mixup', 'mixup_hidden']:
            nat_loss = loss_ce(outputs, targets_reweighted.detach())
            adv_loss = loss_ce(outputs_adv, targets_reweighted2.detach())
            loss = (nat_loss + adv_loss) / 2.0

        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()

        optimizer.step()

        train_loss = loss.item()

        duration = time.time() - start_time
        if batch_idx % args.log_step == 0:
            adv_acc = get_acc(outputs_adv, targets)
            iterator.set_description(str(adv_acc))
            nat_acc = get_acc(outputs, targets)

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
