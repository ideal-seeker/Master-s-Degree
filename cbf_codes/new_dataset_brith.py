import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pandas as pd
import random
import time
import itertools
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D

def dataset_design(num_samples_safe, num_samples_unsafe):
    # 生成安全数据 (num_samples_safe 行，2列)
    x_data_safe = torch.cat([torch.rand(num_samples_safe, 1) * 2 - 1, torch.rand(num_samples_safe, 1) * 2 - 1], dim=1)
    # 将安全数据转换为列表
    x_data_safe_list = x_data_safe.tolist()

    # 生成不安全数据 (num_samples_unsafe 行，2列)
    x_data_unsafe = []
    while len(x_data_unsafe) < num_samples_unsafe:
        x = torch.rand(1, 1) * 4 - 2  # x in [-2, 2]
        y = torch.rand(1, 1) * 4 - 2  # y in [-2, 2]
        # 去掉安全部分
        if not (-1 <= x.item() <= 1 and -1 <= y.item() <= 1):
            x_data_unsafe.append([x.item(), y.item()])  # 添加为列表形式

    # 返回安全数据、不安全数据及其对应的列表形式
    return x_data_safe_list, x_data_unsafe

num_samples_safe = 32*2000
num_samples_unsafe = 32*2000
x_data_safe, x_data_unsafe= dataset_design(num_samples_safe,num_samples_unsafe)
df1 = pd.DataFrame(x_data_safe, columns=['X', 'Y'])
df1.to_csv('safe_points_square.csv', index=False)

# 将第二个数组保存为 CSV 文件
df2 = pd.DataFrame(x_data_unsafe, columns=['X', 'Y'])
df2.to_csv('unsafe_points_square.csv', index=False)

print("两个数组已保存为 CSV 文件")