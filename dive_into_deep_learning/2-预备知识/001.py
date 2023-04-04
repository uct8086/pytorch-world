import torch


# 2.1 数据操作
# X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# print(X)
# print(Y)

# print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))


# print(X == Y)
# print(X < Y)
# print(X > Y)

# 2.2 数据预处理

import os

# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')


import pandas as pd
data = pd.read_csv(data_file)
# print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean()) # 同一列的均值
# print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
# X, y

print(X, y)