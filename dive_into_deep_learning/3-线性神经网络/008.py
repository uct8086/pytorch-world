# 图像分类数据集2 读取小批量
import torchvision
from torch.utils import data
from torchvision import transforms
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.torch import Timer, get_dataloader_workers, load_data_fashion_mnist


# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式， # 并除以255使得所有像素的数值均在0〜1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
root="../../data", train=False, transform=trans, download=True)

print(len(mnist_train), len(mnist_test))

print(mnist_train[0][0].shape)

batch_size = 256 # 如果变成1，时间会变成 36.62s
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())


if __name__ == '__main__': 
    timer = Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec') # 9.58s


# train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
# if __name__ == '__main__': 
#     for X, y in train_iter:
#         print(X.shape, X.dtype, y.shape, y.dtype)
#         break

  