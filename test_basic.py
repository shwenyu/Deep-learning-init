import torch
ones = torch.ones(2, 3)
print(ones) # 输出: tensor([[1., 1., 1.],
#                [1., 1., 1.]])
zeros = torch.zeros(2, 3)
print(zeros) # 输出: tensor([[0., 0., 0.],
#                [0., 0., 0.]])
randn = torch.randn((2, 3), dtype=torch.float32)
# 注意：randn生成的张量中的数值是随机的，每次运行可能会不同
# 例如：tensor([[ 0.1234, -0.5678, 0.9101],
#                [-0.2345, 0.6789, -0.1234]])
print(randn) # 输出: tensor([[ 0.1234, -0.5678, 0.9101],
#                [-0.2345, 0.6789, -0.1234]])
randint = torch.randint(0, 10, (2, 3))
print(randint) # 输出: tensor([[3, 7, 2],
#                [5, 1, 8]])    
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.add(a, b)  # 张量加法
print(c)  # 输出: tensor([[ 6,  8],
#                [10, 12]])
d = torch.sub(a, b)  # 张量减法
print(d)  # 输出: tensor([[-4, -4],
#                [-4, -4]]) 
a += 100 # 张量加法赋值
print(a)  # 输出: tensor([[101, 102],
#                [103, 104]])
c = torch.arange(0, 10)  # 创建一个从0到10，步长为2的张量
print(c)  # 输出: tensor([0, 2, 4, 6, 8])
d = torch.rand(size=(10,10,10))  # 创建一个10x10x10的随机张量
print(d)  # 输出: torch.Size([10, 10, 10])
print(d.dtype) # 输出: cpu (如果在GPU上运行，则输出cuda:0)
e = c+d
print(e.shape)
ten_zeros = torch.zeros_like(c) # 创建一个与c形状相同的全零张量
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype = None,
                               device = None,
                               requires_grad=False)
float_16_tensor = float_32_tensor.type(torch.float16) # 将float_32_tensor转换为float16类型