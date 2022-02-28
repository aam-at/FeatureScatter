import os
import random
import sys

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import get_lambda, mixup_process, to_one_hot

class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 initial_channels,
                 num_classes):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3,
                               initial_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block,
                                       initial_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       initial_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       initial_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       initial_channels * 8,
                                       num_blocks[3],
                                       stride=2)
        self.linear = nn.Linear(initial_channels * 8 * block.expansion,
                                num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def compute_h1(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        return out

    def compute_h2(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward(self, x, target=None, mixup=False, mixup_hidden=False, mixup_alpha=None):
        if mixup_hidden:
            layer_mix = random.randint(0, 2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        out = x

        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype("float32")).cuda()
            lam = Variable(lam)

        if target is not None:
            target_reweighted = to_one_hot(target, self.num_classes)

        if layer_mix == 0:
            out, target_reweighted = mixup_process(out,
                                                   target_reweighted,
                                                   lam=lam)

        out = self.conv1(out)
        out = self.layer1(out)

        if layer_mix == 1:
            out, target_reweighted = mixup_process(out,
                                                   target_reweighted,
                                                   lam=lam)

        out = self.layer2(out)

        if layer_mix == 2:
            out, target_reweighted = mixup_process(out,
                                                   target_reweighted,
                                                   lam=lam)

        out = self.layer3(out)

        if layer_mix == 3:
            out, target_reweighted = mixup_process(out,
                                                   target_reweighted,
                                                   lam=lam)

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if target is not None:
            return self.linear(out), target_reweighted, out
        else:
            return self.linear(out), out


def preactresnet18(num_classes=10, dropout=False):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], 64, num_classes)
