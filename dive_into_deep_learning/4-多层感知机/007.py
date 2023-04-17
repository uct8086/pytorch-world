# 梯度消失/梯度爆炸

import torch
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import plot

if __name__ == "__main__":
    # 梯度消失
    # x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    # y = torch.sigmoid(x)
    # y.backward(torch.ones_like(x))
    # plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
    #         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
    # plot.show()

    # 梯度爆炸
    M = torch.normal(0, 1, size=(4,4)) 
    print('一个矩阵 \n',M)
    for i in range(100):
        M = torch.mm(M,torch.normal(0, 1, size=(4, 4))) 
    print('乘以100个矩阵后\n', M)