import torch
import ssl
import torchvision
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import set_figsize, Image, plt, show_images

ssl._create_default_https_context = ssl._create_unverified_context

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale=scale)

if __name__ == '__main__':
    all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
    show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
 

