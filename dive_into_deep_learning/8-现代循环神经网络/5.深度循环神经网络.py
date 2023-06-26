import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import load_data_time_machine, try_gpu, train_ch8, RNNModel


if __name__ == '__main__':

    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    device = try_gpu()
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    num_epochs, lr = 500, 2
    train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)