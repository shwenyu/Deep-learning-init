import torch
tensor_a = torch.rand(3,4)
tensor_b = torch.rand(3,4)
tensor_b_t = tensor_b.T
tensor_c = torch.matmul(tensor_a, tensor_b_t) # 矩阵乘法
print(tensor_c.shape)  # 输出: torch.Size([3, 3])
tensor_d = torch.mm(tensor_a, tensor_b_t)  # 矩阵乘法
print(tensor_d.shape)  # 输出: torch.Size([3, 3])
tensor_e = tensor_a @ tensor_b_t  # 矩阵乘法
print(tensor_e.shape)  # 输出: torch.Size([3, 3])
tensor_f = torch.bmm(tensor_a.unsqueeze(0), tensor_b_t.unsqueeze(0))  # 批量矩阵乘法
print(tensor_f.shape)  # 输出: torch.Size([1, 3, 3])

x = torch.arange(10,101,10) # 创建一个从10到100，步长为10的张量
print(torch.mean(x.type(torch.float))) # 输出: tensor(55.)
print(x.argmax()) # 输出: tensor(9)  # 最大值的索引
print(x.argmin()) # 输出: tensor(0)  # 最小值的索
x_unsqueeze = x.unsqueeze(1)  # 在第0维增加一个维度
print(x_unsqueeze)  # 输出: torch.Size([1, 10])
x_stack = torch.hstack([x, x+1, x+2]) 
print(x_stack)  # 输出: torch.Size([3, 10])
x_new = x.reshape(2, 5)  
x_new[0,0] = 100  # 修改张量的第一个元素
print(x_new.is_contiguous(), x_new.T.is_contiguous())     
y = torch.rand(2, 1, 2, 1, 4)
g = torch.squeeze(y, (-1, -2))
g = torch.unsqueeze(y, 1)
print(g)
print(is_mps_available := torch.backends.mps.is_available())
device = "mps" if torch.backends.mps.is_available() else "cpu"


abc = torch.rand(3,3)
abc = abc.to(device)
print(torch.cuda.device_count(), device)