# %%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

#%% Scalar
scalar = torch.tensor(7)
scalar.item()

#%% Vector
vector = torch.tensor([7, 7])
vector.ndim

#%% Matrix
MATRIX = torch.tensor([[7, 8], [9, 10]])
MATRIX.ndim
MATRIX[1]
MATRIX.shape

#%% TENSOR
TENSOR = torch.tensor([[[[[[10, 20, 30],
                        [40, 50, 60],
                        [26, 3, 110]]]]]])

TENSOR
TENSOR.ndim
TENSOR.shape