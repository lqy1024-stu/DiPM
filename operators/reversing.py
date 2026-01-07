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

# 加载两个 LoRA 模型参数
lora_model1 = ""
lora_model2 = ""

lora_weights1 = load_safetensors_adapter(lora_model1)
lora_weights2 = load_safetensors_adapter(lora_model2)

cos_sims = []
l2_norms = []
layer_names = []

for name in lora_weights1.keys():
    if name in lora_weights2 and "lora_A" in name:
    # if name in lora_weights2:
        # 计算幅度差异
        diff = lora_weights1[name] - lora_weights2[name]
        l2_norm = torch.norm(diff, p=2).item()
        l2_norms.append(l2_norm)

        # 矩阵平展为向量
        w1 = lora_weights1[name].flatten()
        w2 = lora_weights2[name].flatten()
        # 单位向量归一化
        w1_unit = w1 / w1.norm(p=2)
        w2_unit = w2 / w2.norm(p=2)
        # 计算余弦相似度
        cos_sim = torch.dot(w1_unit, w2_unit).item()
        cos_sims.append(cos_sim)
        layer_names.append(name)

# 转换为NumPy数组
l2_norms = np.array(l2_norms)
cos_sims = np.array(cos_sims)
outliers = (cos_sims >= -1)

for i in np.where(outliers)[0]:
    name = layer_names[i]
    if "lora_A" in name:
        w1 = lora_weights1[name].flatten()
        w2 = lora_weights2[name].flatten()
        new_w1 = - (w2 / w2.norm()) * w1.norm()
        # 恢复原形状
        lora_weights1[name] = new_w1.view_as(lora_weights1[name])


# 保存新的 lora_weights1
save_file(lora_weights1, "path")
