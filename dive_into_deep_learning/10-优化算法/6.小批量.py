import numpy as np
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import get_data_ch11, train_ch11, set_figsize, plot, plt, train_concise_ch11

def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()

def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

if __name__ == '__main__':
    
    # gd_res = train_sgd(1, 1500, 10)

    # sgd_res = train_sgd(0.005, 1)

    # mini1_res = train_sgd(0.4, 100)

    # mini2_res = train_sgd(.05, 10)

    # loss: 0.250, 0.038 sec/epoch
    # loss: 0.243, 0.052 sec/epoch
    # loss: 0.245, 0.003 sec/epoch
    # loss: 0.244, 0.008 sec/epoch

    data_iter, _ = get_data_ch11(10)
    trainer = torch.optim.SGD
    train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
    
 