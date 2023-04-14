# 激活函数
import torch
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.torch import plot

if __name__ == '__main__':
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    # ReLU函数，修正线性单元
    # y = torch.relu(x)
    # plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

    # # 求导数
    # y.backward(torch.ones_like(x), retain_graph=True)
    # plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
 
    # sigmoid 挤压函数
    # y = torch.sigmoid(x)
    # plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
 
    # 求导
    # 清除以前的梯度, 这里不需要清除
    # x.grad.zero_()
    # y.backward(torch.ones_like(x),retain_graph=True)
    # plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

    # tanh(双曲正切)函数
    y = torch.tanh(x)
    # plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
    # 求导
    # 清除以前的梯度, 这里不需要清除
    # x.grad.data.zero_() 
    y.backward(torch.ones_like(x),retain_graph=True)
    plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
 