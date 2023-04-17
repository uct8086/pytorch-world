# 高维线性回归
import numpy as np
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import evaluate_loss, load_array, Animator, linreg, squared_loss, sgd, synthetic_data

# 初始化权重和偏置
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    print(w, b)
    return [w, b]

# 定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 训练函数
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个⻓度为batch_size的向量 
            l = loss(net(X), y) + lambd * l2_penalty(w) 
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
                animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss))) 
    # norm权重归一化
    print('w的L2范数是:', torch.norm(w).item())

# 训练函数高维线性回归简洁实现
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none') 
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
            {"params":net[0].weight,'weight_decay': wd},
            {"params":net[0].bias}], lr=lr)
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                        (evaluate_loss(net, train_iter, loss),
                        evaluate_loss(net, test_iter, loss)))
    print('w的L2范数:', net[0].weight.norm().item())

if __name__ == "__main__":

    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = synthetic_data(true_w, true_b, n_train)
    train_iter = load_array(train_data, batch_size)
    test_data = synthetic_data(true_w, true_b, n_test)
    test_iter = load_array(test_data, batch_size, is_train=False)

    # train(lambd=0)
    # train(lambd=3)
    train_concise(10)