import torch
import math
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import show_trace_2d, train_2d


def f(x1, x2): # 目标函数 
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2): # 目标函数的梯度 
    return 2 * x1, 4 * x2

def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 模拟有噪声的梯度
    g1 += torch.normal(0.0, 1, (1,)) 
    g2 += torch.normal(0.0, 1, (1,)) 
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
    return 1

# 指数衰减
def exponential_lr():
    # 在函数外部定义，而在内部更新的全局变量
    global t
    t += 1
    return math.exp(-0.1 * t)

# 多项式衰减
def polynomial_lr():
    # 在函数外部定义，而在内部更新的全局变量 
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

if __name__ == '__main__':

    eta = 0.1
    # lr = constant_lr # 常数学习速度
    # show_trace_2d(f, train_2d(sgd, steps=50, f_grad=f_grad, init_tensor=True))
    
    # 动态学习率
    # t=1
    # lr = exponential_lr
    # show_trace_2d(f, train_2d(sgd, steps=1000, f_grad=f_grad, init_tensor=True))

    t=1
    lr = polynomial_lr
    show_trace_2d(f, train_2d(sgd, steps=50, f_grad=f_grad, init_tensor=True))