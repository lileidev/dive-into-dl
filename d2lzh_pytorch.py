import random
import sys
from IPython import display
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('TkAgg')

def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

def linreg(X, w, b):
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 
                   'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_data_fashion_mnist(batch_size):
    batch_size = 256
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=True, download=False, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=False, download=False, transform=transforms.ToTensor())


    train_iter = torch.utils.data.DataLoader(mnist_train,
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    
    return train_iter, test_iter