import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import torch.backends.cudnn
from datetime import datetime
import os


def load_data(name, train):
    if train:
        if name == 'MNIST':
            dataset = MNIST('../data', download=True, train=True)
            return dataset.data / 255.0, dataset.targets
        elif name == 'FashionMNIST':
            dataset = FashionMNIST('../data', download=True, train=True)
            return dataset.data / 255.0, dataset.targets
        elif name == 'CIFAR10':
            dataset = CIFAR10('../data', download=True, train=True)
            return torch.tensor(dataset.data).permute(0, 3, 1, 2), torch.tensor(dataset.targets)
        elif name == 'CIFAR100':
            dataset = CIFAR100('../data', download=True, train=True)
            return torch.tensor(dataset.data).permute(0, 3, 1, 2), torch.tensor(dataset.targets)
        if name == 'HalfMNIST':
            dataset = MNIST('../data', download=True, train=True)
            indexes = dataset.targets < 5
            return dataset.data[indexes] / 255.0, dataset.targets[indexes]
        else:
            raise NameError('Invalid dataset name.')
    else:
        if name == 'MNIST':
            dataset = MNIST('../data', download=True, train=False)
            return dataset.data / 255.0, dataset.targets
        elif name == 'FashionMNIST':
            dataset = FashionMNIST('../data', download=True, train=False)
            return dataset.data / 255.0, dataset.targets
        elif name == 'CIFAR10':
            dataset = CIFAR10('../data', download=True, train=False)
            return torch.tensor(dataset.data).permute(0, 3, 1, 2), torch.tensor(dataset.targets)
        elif name == 'CIFAR100':
            dataset = CIFAR100('../data', download=True, train=False)
            return torch.tensor(dataset.data).permute(0, 3, 1, 2), torch.tensor(dataset.targets)
        if name == 'HalfMNIST':
            dataset = MNIST('../data', download=True, train=False)
            indexes = dataset.targets < 5
            return dataset.data[indexes] / 255.0, dataset.targets[indexes]
        else:
            raise NameError('Invalid dataset name.')


def shuffle_class_order(continuum):
    labels = continuum.labels
    new_labels = torch.zeros_like(labels)
    classes = torch.unique(labels)
    permutation = torch.randperm(classes.shape[0])
    for i, j in enumerate(permutation):
        new_labels[labels == i] = j
    order = torch.argsort(new_labels)
    continuum.inputs, continuum.labels = continuum.inputs[order], continuum.labels[order]


def my_barplot(x, y, title=None):
    # Deal with missing classes
    if x[-1] >= 10:
        n_classes = 100
    elif x[-1] >= 5:
        n_classes = 10
    else:
        n_classes = 5
    for i in range(n_classes):
        if i < len(x):
            if i != x[i]:
                x = x[:i] + [i] + x[i:]
                y = y[:i] + [0] + y[i:]
        else:
            x.append(i)
            y.append(0)

    # Draw the bar plot
    plt.bar(x, y, alpha=1, edgecolor='black')
    plt.xlabel('Class ID')
    plt.xticks(x)
    plt.yticks([])
    plt.ylim([0, max(y) * 1.2])
    ax = plt.gca()
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    if title is not None:
        plt.title(title)
    font = {'size': 9, 'weight': 'bold'}
    gap = max(y) * 0.02
    for i, elem in enumerate(y):
        plt.text(i, elem + gap, str(int(elem)), fontdict=font, ha='center')


def confidence_interval(data_list):
    array = np.array(data_list)
    mean = float(np.mean(array))
    std = float(np.std(array))
    return f'{mean:.1f} \\pm {std:.1f}'


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def unique_name(prefix=''):
    now = datetime.now()
    datetime_string = now.strftime("%d%b%Y_%H%M%S%f")
    return prefix + datetime_string


def create_results_directory(dir_name):
    path = f'../results/' + dir_name + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path
