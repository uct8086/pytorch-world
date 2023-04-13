import torch
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.torch import load_data_fashion_mnist, accuracy, evaluate_accuracy, sgd, train_ch3, predict_ch3

if __name__ == '__main__': 
    batch_size = 256
    # 每个样本都是28 × 28的图像。展平每个图像，把它们看作⻓度为784的向量,数据集有10个类别
    train_iter, test_iter = load_data_fashion_mnist(batch_size) # 加载图片数据集

    num_inputs = 784
    num_outputs = 10
    # 权重将构成一个784 × 10的矩阵，偏置将构成一个1 × 10的行向量
    # 权重
    # normal方法是正态分布, size的意思是 784行，10列的一个矩阵
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) 
    # 偏置
    b = torch.zeros(num_outputs, requires_grad=True)

    # print(W, b)

    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # keepdim就是保持形状
    # print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))

    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition # 这里应用了广播机制

    X = torch.normal(0, 1, (2, 5)) # 2行5列的一个正态分布矩阵
    # print(X)
    # 通过softmax回归方法得到全都是正值的行值相加都为1的矩阵
    X_prob = softmax(X)
    print(X_prob, X_prob.sum(1)) # 这里的sum意思就是保留列的维度相加 0行，1列

    # softmax回归模型
    def net(X):
        return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

    # y 是标签，也就是目标
    y = torch.tensor([0, 2]) 
    # 2样本在3个类别中的预测概率
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    # 使用y作为y_hat中概率的索引
    print(y_hat[[0, 1], y])

    # 交叉熵损失函数
    def cross_entropy(y_hat, y):
        return - torch.log(y_hat[range(len(y_hat)), y])
    
    print(cross_entropy(y_hat, y))


    # 分类精度预测
    print(accuracy(y_hat, y) / len(y))

    print(evaluate_accuracy(net, test_iter))

    # 训练
    lr = 0.1
    def updater(batch_size):
        return sgd([W, b], lr, batch_size)
    #训练10轮
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    # 输出预测图像
    predict_ch3(net, test_iter)
 