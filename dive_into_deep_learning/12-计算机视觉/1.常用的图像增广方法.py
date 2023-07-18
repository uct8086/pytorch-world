import torch
import torchvision
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import set_figsize, Image, plt, show_images


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale=scale)

if __name__ == '__main__':
    set_figsize()
    img = Image.open('../data/img/深圳湾.jpg')
    # plt.imshow(img);
    # plt.show()

    # apply(img, torchvision.transforms.RandomHorizontalFlip())
    # apply(img, torchvision.transforms.RandomVerticalFlip())

    # 裁剪

    shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    # apply(img, shape_aug)
 
    # 改变颜色

    # apply(img, torchvision.transforms.ColorJitter(
    # brightness=0.5, contrast=0, saturation=0, hue=0))

    # apply(img, torchvision.transforms.ColorJitter(
    #  brightness=0, contrast=0, saturation=0, hue=0.5))

    color_aug = torchvision.transforms.ColorJitter(
     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # apply(img, color_aug)

    augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
    apply(img, augs)
 
 
 