import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file, save_file
from scipy.spatial.distance import cosine
from scipy.stats import linregress
import torch.nn.functional as F

# 加载 .safetensors 文件
def load_safetensors_adapter(file_path):
    tensors = load_file(file_path)
    return tensors

# 加载两个 LoRA 模型参数
lora_model1 = ''
lora_model2 = ''

lora_weights1 = load_safetensors_adapter(lora_model1)
lora_weights2 = load_safetensors_adapter(lora_model2)

def induce(w1, w2, theta):
    w1 = w1.float()
    w2 = w2.float()
    # 单位向量
    u1 = w1 / (w1.norm() + 1e-8)
    u2 = w2 / (w2.norm() + 1e-8)
    new_w1 = (1-theta)*u1 + theta*u2
    new_u1 = new_w1 / (new_w1.norm() + 1e-8)
    new_vec = new_u1 * w2.norm()
    return new_vec 

for name in lora_weights1.keys():
    if "lora_B" in name:
        w1 = lora_weights1[name].flatten()
        w2 = lora_weights2[name].flatten()
        new_w1 = induce(w1, w2, 10)
        # 恢复原形状
        lora_weights1[name] = new_w1.view_as(lora_weights1[name])

# 保存新的 lora_weights1
save_file(lora_weights1, "path")
