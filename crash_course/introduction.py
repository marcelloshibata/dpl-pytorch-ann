# %%
import torch

torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)

# %% Tensors
import numpy as np

arr = np.array([[1,2,3], [4,5,6]])
arr
#%%
tensor = torch.Tensor([[1,2,3], [4,5,6]])
tensor = torch.from_numpy(arr)
tensor

np.random.random((2,4))
torch.rand((2,4))

arr.shape
arr.dtype

tensor.shape
tensor.dtype
tensor.device
# arr.to_device('cuda') não funciona pois numpy aceita apenas cpu.

tensor = tensor.to('cuda')
tensor.sum()

tensor.cpu().numpy() # converter para array numpy

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'