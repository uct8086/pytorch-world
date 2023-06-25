import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import load_data_time_machine, RNNModelScratch, try_gpu, predict_ch8, grad_clipping, train_ch8


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device) # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device) # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
            param.requires_grad_(True)
    return params

# 返回隐状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    # inputs的形状:(时间步数量，批量大小，词表大小) 
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状:(批量大小，词表大小)
    for X in inputs:
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

if __name__ == '__main__':

    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    # F.one_hot(torch.tensor([0, 2]), len(vocab))
    X = torch.arange(10).reshape((2, 5))
    # F.one_hot(X.T, 28).shape


    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(), get_params,
                        init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], try_gpu())
    Y, new_state = net(X.to(try_gpu()), state)
    print(Y.shape, len(new_state), new_state[0].shape)

    # # 预热+预测
    # print(predict_ch8('time traveller ', 10, net, vocab, try_gpu()))
    # # 训练
    num_epochs, lr = 500, 1
    # train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu())

    # 用随机抽样方法的结果 

    net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(), get_params,
                      init_rnn_state, rnn)
    train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu(),
            use_random_iter=True)