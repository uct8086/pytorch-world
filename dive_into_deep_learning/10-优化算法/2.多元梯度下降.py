import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import show_trace_2d, train_2d


def f_2d(x1, x2): # 目标函数 
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2): # 目标函数的梯度 
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)


if __name__ == '__main__':

    eta = 0.1
    show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))