import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import load_data_time_machine, RNNModel, try_gpu, predict_ch8, train_ch8



if __name__ == '__main__':

    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    # 初始化隐状态
    state = torch.zeros((1, batch_size, num_hiddens))

    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, state_new.shape)

    # 训练与预测

    device = try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    # predict_ch8('time traveller', 10, net, vocab, device)

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)