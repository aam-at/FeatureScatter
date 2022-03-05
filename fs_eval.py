from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import datetime

from tqdm import tqdm
from models import *
import utils

from attack_methods import Attack_None, Attack_PGD

from utils import softCrossEntropy, CWLoss

parser = argparse.ArgumentParser(
    description='Feature Scattering Adversarial Training')

parser.register('type', 'bool', utils.str2bool)

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--attack', default=True, type='bool', help='attack')
parser.add_argument('--data_dir', type=str, help='path where to store the data')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass')

parser.add_argument('--attack_method',
                    default='pgd',
                    type=str,
                    help='adv_mode (natural, pdg or cw)')
parser.add_argument('--attack_method_list', type=str)

parser.add_argument('--log_step', default=7, type=int, help='log_step')

# dataset dependent
parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')  # concat cascade
parser.add_argument('--batch_size_test',
                    default=100,
                    type=int,
                    help='batch size for testing')
parser.add_argument('--image_size', default=32, type=int, help='image size')
parser.add_argument('--model', type=str, default='preactresnet18', help='model name', choices=['wrn28_10', 'preactresnet18'])

args = parser.parse_args()


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

log = open(os.path.join(args.model_dir, 'log_test.txt'), 'w')

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
elif args.dataset == 'mnist':
    print_log('----------mnist---------', log)
    args.num_classes = 10
    args.image_size = 28

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

# Data
print_log('==> Preparing data..', log)

if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

if args.dataset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root=args.data_dir,
                                           train=False,
                                           download=True,
                                           transform=transform_test)
elif args.dataset == 'cifar100':
    testset = torchvision.datasets.CIFAR100(root=args.data_dir,
                                            train=False,
                                            download=True,
                                            transform=transform_test)

elif args.dataset == 'svhn':
    testset = torchvision.datasets.SVHN(root=args.data_dir,
                                        split='test',
                                        download=True,
                                        transform=transform_test)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch_size_test,
                                         shuffle=False,
                                         num_workers=2)

print_log('==> Building model..', log)
if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
    print_log(args.model, log)
    if args.model == 'wrn28_10':
        net = wrn28_10(num_classes=args.num_classes, widen_factor=10, per_img_std=True)
    elif args.model == 'preactresnet18':
        net = preactresnet18(num_classes=args.num_classes, per_img_std=True)

net = net.to(device)

# configs
config_natural = {'train': False}

config_fgsm = {
    'targeted': False,
    'epsilon': 8.0,
    'num_steps': 1,
    'step_size': 8.0,
    'random_start': False
}

config_pgd = {
    'targeted': False,
    'epsilon': 8.0,
    'num_steps': 20,
    'step_size': 2.0,
    'random_start': False,
    'loss_func': torch.nn.CrossEntropyLoss(reduction='none')
}

config_cw = {
    'targeted': False,
    'epsilon': 8.0,
    'num_steps': 20,
    'step_size': 2.0,
    'random_start': False,
    'loss_func': CWLoss(args.num_classes)
}


def test(epoch, net, attack):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    iterator = tqdm(testloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        pert_inputs = attack(inputs, targets).detach()

        pert_outputs, _ = net(pert_inputs, targets)

        loss = criterion(pert_outputs, targets)
        test_loss += loss.item()

        duration = time.time() - start_time

        _, predicted = pert_outputs.max(1)
        batch_size = targets.size(0)
        total += batch_size
        correct_num = predicted.eq(targets).sum().item()
        correct += correct_num
        iterator.set_description(
            str(predicted.eq(targets).sum().item() / targets.size(0)))

        if batch_idx % args.log_step == 0:
            print_log(
                "step %d, duration %.2f, test  acc %.2f, avg-acc %.2f, loss %.2f"
                % (batch_idx, duration, 100. * correct_num / batch_size,
                   100. * correct / total, test_loss / total), log)

    acc = 100. * correct / total
    print_log('Val acc:', acc, log)
    return acc


attack_list = args.attack_method_list.split('-')
attack_num = len(attack_list)

for attack_idx in range(attack_num):

    args.attack_method = attack_list[attack_idx].upper()
    print_log(f"-----{args.attack_method} adv mode -----", log)

    if args.attack_method == 'NATURAL':
        # config is only dummy, not actually used
        attack = Attack_None(net, config_natural)
    elif args.attack_method == 'FGSM':
        attack = Attack_PGD(net, config_fgsm)
    elif args.attack_method.startswith('PGD'):
        num_steps = int(args.attack_method.replace('PGD', ''))
        attack = Attack_PGD(net, config_pgd.update({'num_steps': num_steps}))
    elif args.attack_method.startswith('CW'):
        num_steps = int(args.attack_method.replace('CW', ''))
        attack = Attack_PGD(net, config_cw.update({'num_steps': num_steps}))
    else:
        raise Exception(
            'Should be a valid attack method. The specified attack method is: {}'
            .format(args.attack_method))

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume and args.init_model_pass != '-1':
        # Load checkpoint.
        print_log('==> Resuming from checkpoint..', log)
        f_path_latest = os.path.join(args.model_dir, 'latest')
        f_path = os.path.join(args.model_dir,
                              ('checkpoint-%s' % args.init_model_pass))
        if not os.path.isdir(args.model_dir):
            print_log('train from scratch: no checkpoint directory or file found', log)
        elif args.init_model_pass == 'latest' and os.path.isfile(
                f_path_latest):
            checkpoint = torch.load(f_path_latest)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            print_log('resuming from epoch %s in latest' % start_epoch, log)
        elif os.path.isfile(f_path):
            checkpoint = torch.load(f_path)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            print_log('resuming from epoch %s' % start_epoch, log)
        elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
            print_log('train from scratch: no checkpoint directory or file found', log)

    criterion = nn.CrossEntropyLoss()

    test(0, net)
