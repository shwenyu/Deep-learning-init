import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
'''
# 创建张量
x = torch.randn(2, 3)  # 2x3的随机张量
print(x.shape)  # torch.Size([2, 3])

# 张量运算
y = torch.randn(3, 4)
z = torch.mm(x, y)  # 矩阵乘法
print(z.shape)  # torch.Size([2, 4])

# 自动求导
x = torch.randn(2, 3, requires_grad=True)
y = x.sum()
y.backward()  # 反向传播
print(x.grad)  # 梯度
'''
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNet(input_size=784, hidden_size=128, output_size=10)
print(model)
