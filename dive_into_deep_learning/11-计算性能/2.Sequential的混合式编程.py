import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import Benchmark
# 生产网络的工厂模式 
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net


if __name__ == '__main__':
   
    
    x = torch.randn(size=(1, 512))
    net = get_net()
    print(net(x))

    net = torch.jit.script(net)
    print(net(x))

    net = get_net()
    with Benchmark('无torchscript'):
        for i in range(1000): net(x)
    net = torch.jit.script(net)
    with Benchmark('有torchscript'): 
        for i in range(1000): net(x)

    net.save('my_mlp')

    # 命令行输入下面代码查看
    # !ls -lh my_mlp*