# 从0开始实现多层感知机
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import load_data_fashion_mnist, train_ch3, predict_ch3

# 手动实现ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1) # 这里“@”代表矩阵乘法 
    return (H@W2 + b2)

if __name__ == '__main__': 
    batch_size = 256
    # 获取数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    # 特征，分类，隐藏单元
    num_inputs, num_outputs, num_hiddens = 784, 10, 128
    # 隐藏层权重
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    # 隐藏层偏置
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    # 输出层权重
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    # 输出层偏置
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]
    # 这里还是用交叉熵损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 训练
    num_epochs, lr = 10, 0.1
    # SGD随机梯度下降
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    predict_ch3(net, test_iter)