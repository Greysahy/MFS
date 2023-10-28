import torch

N = 4
x = torch.randint(1, 10, (4, 4))
y = torch.gather(x, dim=1)
print(x)
print(y)