# %%
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

f = 3 * a**3 - b**2
f.backward(gradient=torch.tensor([1,1]))
print(a.grad)
print(b.grad)