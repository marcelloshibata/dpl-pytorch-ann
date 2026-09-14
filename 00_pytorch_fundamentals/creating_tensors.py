#%%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Random Tensors
# Tensor aleatório de tamanho (3, 4)
random_tensor = torch.rand(3, 4)
random_tensor

#%% Tensor aleatório com forma similar a um tensor de imagem
random_image_size_tensor = torch.rand(size=(224, 224, 3)) # altura, largura, canais de cor (R,G,B)
random_image_size_tensor.shape, random_image_size_tensor.ndim

#%% Tensors com 0 e 1
zero = torch.zeros(size=(3, 4))
ones = torch.ones(size=(3, 4))