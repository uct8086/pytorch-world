import torch
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import set_figsize, plot


def f(x): # O目标函数
    return torch.cosh(c * x)
def f_grad(x): # 目标函数的梯度
    return c * torch.sinh(c * x) 
def f_hess(x): # 目标函数的Hessian
    return c**2 * torch.cosh(c * x)
def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    set_figsize()
    plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

if __name__ == '__main__':

    c = torch.tensor(0.5) 
    
    show_trace(newton(), f)