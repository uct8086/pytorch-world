# 图像分类数据集
import torchvision
from torch.utils import data
from torchvision import transforms
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.torch import get_fashion_mnist_labels, show_images, use_svg_display, plot

use_svg_display()

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式， # 并除以255使得所有像素的数值均在0〜1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
root="../../data", train=False, transform=trans, download=True)

print(len(mnist_train), len(mnist_test))

print(mnist_train[0][0].shape)

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

  