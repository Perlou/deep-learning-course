import torch

torch.manual_seed(42)

x = torch.randn(3, 4, 5)

print(x)
print(x.shape)  # torch.Size([3, 4, 5])
print(x.size())  # torch.Size([3, 4, 5])
print(x.size(0))  # 3 (第0维大小)
print(x.dtype)  # torch.float32
print(x.device)  # cpu
print(x.ndim)  # 3
print(x.numel())  # 60
print(x.requires_grad)  # False
