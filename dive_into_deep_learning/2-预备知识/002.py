import torch

# A = torch.arange(20).reshape(5, 4) # 定义一个矩阵

# print(A)
# print(A.T) # 矩阵的转置

# B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])

# print(B == B.T) # 对称矩阵

# 多阶张量，可以从后往前算，列，行，其他
# X = torch.arange(24).reshape(2, 3, 4)

# print(X)

# 求和

# x = torch.arange(4, dtype=torch.float32)

# print(x, x.sum())

# 降维
# A = torch.arange(12, dtype=torch.float32).reshape(4, 3)
# print(A)
# print(A, A.shape, A.sum())

# A_sum_axis0 = A.sum(axis=0) # 轴0 相加
# print(A_sum_axis0, A_sum_axis0.shape)

# A_sum_axis1 = A.sum(axis=1) # 轴1 相加
# print(A_sum_axis1, A_sum_axis1.shape)

# A.sum(axis=[0, 1]) # 结果和A.sum()相同

# # 均值

# print(A.mean(), A.sum() / A.numel())

# 非降维求和

# sum_A = A.sum(axis=1, keepdims=True)

# print(sum_A)

# print( A / sum_A)

# # 不沿任何轴降低输入张量维度
# # 一行一行往下累加
# print(A.cumsum(axis=0))

# 点积
A = torch.arange(20, dtype = torch.float32).reshape(5, 4) # 定义一个矩阵
y = torch.ones(4, dtype = torch.float32)
x = torch.arange(4, dtype = torch.float32)

# 点积，先相同位置相乘，再各元素相加
# print(x, y, torch.dot(x, y)) # 等同于 torch.sum(x * y)

# 矩阵向量积
# print(A)
# print(x)
# print(A.shape, x.shape, torch.mv(A, x)) # mv方法就是没地方矩阵的向量积, A的列维数(沿轴1的⻓度)必须与x的维数(其⻓度)相同

# 矩阵乘法

B = torch.ones(4, 3)
# print(B)
# print(torch.mm(A, B))

# 向量范数

u = torch.tensor([3.0, -4.0]) # 就是 3 * 3 + （-4 * -4） ，然后开平方根
 
print(u)
print(torch.norm(u))