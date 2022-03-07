from functools import reduce
from operator import __or__

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def load_data(batch_size,
              workers,
              dataset,
              data_target_dir,
              valid_labels_per_class=500):
    # copied from GibbsNet_pytorch/load.py
    if dataset == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
        ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Lambda(
                                             lambda x: x.mul(255)),
                                         ])
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(data_target_dir,
                                      train=True,
                                      transform=train_transform,
                                      download=True)
        test_data = datasets.CIFAR10(data_target_dir,
                                     train=False,
                                     transform=test_transform,
                                     download=True)
        num_classes = 10
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(data_target_dir,
                                       train=True,
                                       transform=train_transform,
                                       download=True)
        test_data = datasets.CIFAR100(data_target_dir,
                                      train=False,
                                      transform=test_transform,
                                      download=True)
        num_classes = 100
    elif dataset == "svhn":
        train_data = datasets.SVHN(data_target_dir,
                                   split="train",
                                   transform=train_transform,
                                   download=True)
        test_data = datasets.SVHN(data_target_dir,
                                  split="test",
                                  transform=test_transform,
                                  download=True)
        num_classes = 10
    else:
        assert False, "Do not support dataset : {}".format(dataset)

    n_labels = num_classes

    def get_sampler(labels, n_valid=None):
        # Only choose digits in n_labels
        # n = number of labels per class for training
        # n_val = number of lables per class for validation
        # print type(labels)
        # print (n_valid)
        (indices, ) = np.where(
            reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)

        indices_valid = np.hstack([
            list(filter(lambda idx: labels[idx] == i, indices))[:n_valid]
            for i in range(n_labels)
        ])
        indices_train = np.hstack([
            list(filter(lambda idx: labels[idx] == i, indices))[n_valid:]
            for i in range(n_labels)
        ])
        indices_unlabelled = np.hstack([
            list(filter(lambda idx: labels[idx] == i, indices))[:]
            for i in range(n_labels)
        ])
        # import pdb; pdb.set_trace()
        # print (indices_train.shape)
        # print (indices_valid.shape)
        # print (indices_unlabelled.shape)
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled

    # print type(train_data.train_labels)

    # Dataloaders for MNIST
    if dataset == "svhn":
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(
            train_data.labels, valid_labels_per_class)
    else:
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(
            train_data.targets, valid_labels_per_class)

    labelled = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           shuffle=False,
                                           num_workers=workers,
                                           pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             sampler=valid_sampler,
                                             shuffle=False,
                                             num_workers=workers,
                                             pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             sampler=unlabelled_sampler,
                                             shuffle=False,
                                             num_workers=workers,
                                             pin_memory=True)
    test = torch.utils.data.DataLoader(test_data,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=workers,
                                       pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes
