# 暂退法
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import load_data_fashion_mnist, train_ch3

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    # 这个方法重载了父类的方法，在__call__()方法中会被调用
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs)))) 
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

# 以dropout的概率丢弃张量输入X中的元素
def dropout_layer(X, dropout): 
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃 
    if dropout == 1:
        return torch.zeros_like(X) 
    # 在本情况中，所有元素都被保留 
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

if __name__ == "__main__":

    X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
    print(X)
    print(dropout_layer(X, 0.))
    print(dropout_layer(X, 0.5))
    print(dropout_layer(X, 1.))

    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

    dropout1, dropout2 = 0.2, 0.5
    # 初始实现
    # net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    # 简洁实现
    net = nn.Sequential(nn.Flatten(),
                nn.Linear(784, 256),
                nn.ReLU(),
                # 在第一个全连接层之后添加一个dropout层 
                nn.Dropout(dropout1),
                nn.Linear(256, 256),
                nn.ReLU(),
                # 在第二个全连接层之后添加一个dropout层 
                nn.Dropout(dropout2),
                nn.Linear(256, 10))
       
    net.apply(init_weights)
    # 训练
    num_epochs, lr, batch_size = 10, 0.5, 256
    # 交叉熵损失
    loss = nn.CrossEntropyLoss(reduction='none')
    # 加载训练数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    # 模型，随机梯度下降
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    # 执行
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
 