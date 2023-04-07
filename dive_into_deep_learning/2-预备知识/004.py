import torch

x = torch.arange(4.0, requires_grad=True)

print(x)
print(x.grad) # 梯度

y = 2 * torch.dot(x, x)

print(y)

print(y.backward(), x.grad)

print(x.grad == 4*x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值 
x.grad.zero_()
print(x)
y = x.sum()
print(y)
y.backward()
print(y)
print(x.grad)


# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。 
# # 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y=x*x
# 等价于y.backward(torch.ones(len(x))) 
y.sum().backward()
print(x.grad)