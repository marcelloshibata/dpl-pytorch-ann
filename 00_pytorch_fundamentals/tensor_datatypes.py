#%%
import torch

float_32_tensor = torch.tensor([[[3.0, 6.0, 9.0]]], 
                               dtype=None, # Qual tipo de dado é o tensor (float32, float16)
                               device=None, # Qual dispositivo o tensor está
                               requires_grad=False) 

float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor.dtype

mult = float_16_tensor * float_32_tensor * 32