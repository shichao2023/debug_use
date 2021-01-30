import numpy as np
from collections.abc import Iterable
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import math
from scipy.special import softmax
import torchvision.datasets as datasets

def get_cifar_data(dir):

    list_img = []
    list_label = []
    data_size = 0

    #load cifar in the format as (32,32,3)
    for filename in ['%s/data_batch_%d' % (dir,j) for j in range(1, 6)]:
        with open(filename, 'rb') as fo:
            cifar10 = pickle.load(fo, encoding = 'bytes')
        for i in range(len(cifar10[b"labels"])):
            img = np.reshape(cifar10[b"data"][i], (3,32,32))
            img = np.transpose(img, (1,2,0))
            #img = img.astype(float)
            list_img.append(img)
            
            list_label.append(np.eye(10)[cifar10[b"labels"][data_size%10000]])
            data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]

def get_mnist_data():

    list_img = []
    list_label = []
    data_size = 0

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    for i in range(len(mnist_trainset)):
        img = np.array(mnist_trainset[i][0])
        img = np.pad(img, ((2,2),(2,2)))
        img = np.expand_dims(img, 2).repeat(3, axis=2)
        list_img.append(img)
        list_label.append(np.eye(10)[mnist_trainset[i][1]])
        data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

def get_svhn_data():

    list_img = []
    list_label = []
    data_size = 0

    svhn_trainset = datasets.SVHN(root='./data', split='train', download=True, transform=None)

    for i in range(len(svhn_trainset)):
        list_img.append(np.array(svhn_trainset[i][0]))
        assert list_img[-1].shape == (32, 32, 3)
        list_label.append(np.eye(10)[svhn_trainset[i][1]])
        data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(y_pred, y_actual, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res



def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


if __name__ == '__main__':
    mnist = get_mnist_data()
    svhn = get_svhn_data()
    cifar = get_cifar_data('./data/cifar-10-batches-py/')
    print(mnist[0].shape, svhn[0].shape, cifar[0].shape, mnist[1].shape, svhn[1].shape, cifar[1].shape)