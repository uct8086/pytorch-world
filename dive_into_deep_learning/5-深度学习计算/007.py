# 读写文件

import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


if __name__ == '__main__':

    x = torch.arange(4)
    # 存储张量
    # torch.save(x, '../data/x-file')

    # x2 = torch.load('../data/x-file')
    # print(x2)
    # # 张量列表
    # y = torch.zeros(4)
    # torch.save([x, y],'../data/x-files')
    # x2, y2 = torch.load('../data/x-files')
    # print((x2, y2))
    # # 张量字典
    # mydict = {'x': x, 'y': y}
    # torch.save(mydict, '../data/mydict')
    # mydict2 = torch.load('../data/mydict')
    # print(mydict2)

    # 加载和保存模型参数 
    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)
    torch.save(net.state_dict(), '../data/mlp.params')

    clone = MLP()
    clone.load_state_dict(torch.load('../data/mlp.params'))
    print(clone.eval())
    
    Y_clone = clone(X)
    print(Y_clone == Y)