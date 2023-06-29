import torch
import numpy as np
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import set_figsize, plot


def f(x): # 目标函数 
    return x ** 2

def f_grad(x): # 目标函数的梯度(导数) 
    return 2 * x

# eta 就是指学习率
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) # 这一步是更新参数的关键
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    set_figsize()
    plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

if __name__ == '__main__':

    results = gd(0.2, f_grad)
    print(results)

    show_trace(results, f)