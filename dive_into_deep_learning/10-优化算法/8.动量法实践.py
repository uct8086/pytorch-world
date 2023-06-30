import numpy as np
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import get_data_ch11, train_ch11, train_concise_ch11

def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)
 
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()

def train_momentum(lr, momentum, num_epochs=2):
    train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

if __name__ == '__main__':
    
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    # train_momentum(0.02, 0.5)
    # train_momentum(0.01, 0.9)
    # train_momentum(0.005, 0.9)

    # 简洁实现

    trainer = torch.optim.SGD
    train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
 
    