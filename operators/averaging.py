import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file, save_file
from scipy.spatial.distance import cosine
from scipy.stats import linregress

# 加载 .safetensors 文件
def load_safetensors_adapter(file_path):
    tensors = load_file(file_path)
    return tensors

# 加载LoRA模型参数
lora_model1 = ''
lora_model2 = 's'
lora_model3 = ''
lora_model4 = ''

lora_weights1 = load_safetensors_adapter(lora_model1)
lora_weights2 = load_safetensors_adapter(lora_model2)
lora_weights3 = load_safetensors_adapter(lora_model3)
lora_weights4 = load_safetensors_adapter(lora_model4)

for name in lora_weights1.keys():
    # if "lora_B" in name:
        w1 = lora_weights1[name].flatten()
        w2 = lora_weights2[name].flatten()
        w3 = lora_weights3[name].flatten()
        w4 = lora_weights4[name].flatten()
        # 求平均方向
        w1_unit = w1 / (w1.norm() + 1e-8)
        w2_unit = w2 / (w2.norm() + 1e-8)
        w3_unit = w3 / (w3.norm() + 1e-8)
        w4_unit = w4 / (w4.norm() + 1e-8)
        mean = (w1_unit + w2_unit + w3_unit + w4_unit) / 4
        mean_unit = mean / (mean.norm() + 1e-8)
        w1_new = mean_unit
        lora_weights1[name] = w1_new.view_as(lora_weights1[name])


# 保存新的 lora_weights1
save_file(lora_weights1, "path")
