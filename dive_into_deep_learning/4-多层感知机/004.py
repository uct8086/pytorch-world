# 多项式回归
import math
import numpy as np
import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import evaluate_loss, load_array, Animator, train_epoch_ch3

# train_features 依据的自变量， train_labels 预测的目标 
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    # MSE: Mean Squared Error（均方误差）
    # 含义：均方误差，是预测值与真实值之差的平方和的平均值
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1,1)),
                                    batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1,1)),
                            batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 这里是显示函数
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                    evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

if __name__ == "__main__":

    # 多项式的最⼤阶数
    max_degree = 20
    # 训练和测试数据集⼤⼩
    n_train, n_test = 100, 100
    # 分配⼤量的空间
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    print(true_w)
    # 用正态分布生成特征
    features = np.random.normal(size=(n_train + n_test, 1))
    #  random.shuffle方法，对元素进行重新排序，打乱原有的顺序，返回一个随机序列(当然此处随机序列属于伪随机，即可重现)，该方法的作用类似洗牌。
    np.random.shuffle(features)
    # np.power(a,b)就是求 a的次方，a,b可以是矩阵
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        # gamma(n)=(n-1)! 伽马函数的特性, 阶乘
        # [:, i] 取第i列
        poly_features[:, i] /= math.gamma(i + 1)

    # labels的维度:(n_train+n_test,)
    labels = np.dot(poly_features, true_w)
    # print(np.random.normal(scale=0.1, size=labels.shape))
    labels += np.random.normal(scale=0.1, size=labels.shape)
    # print(labels)
    # NumPy ndarray转换为tensor
    true_w, features, poly_features, labels = [
        torch.tensor(x, dtype=torch.float32)
        for x in [true_w, features, poly_features, labels]
    ]

    # print(features[:2], poly_features[:2, :], labels[:2])
    # print(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
    # 1、从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3! , 取前4列做训练，
    # train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
    # weight: [[ 5.000446   1.1925726 -3.3994446  5.613429 ]]
    # 2、从多项式特征中选择前2个维度，即1和x 
    # train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
    # weight: [[3.3861537 3.623168 ]]
    # 3、从多项式特征中选取所有维度
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)