import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def draw_2d_projection_fixed(file_path):
    """
    绘制2D数据散点图（与9D版本风格保持一致）
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    try:
        df = pd.read_csv(file_path)

        # 自动截取前两列
        if df.shape[1] < 2:
            print("错误：CSV文件至少需要包含2列数据")
            return

        df = df.iloc[:, :2]
        data_array = df.values

        print(f"成功读取数据: {data_array.shape}")

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 维度名
    dim_names = ["D1", "D2"]

    # 创建图形
    plt.figure(figsize=(6, 5))

    # 绘制散点图
    plt.scatter(data_array[:, 0],
                data_array[:, 1],
                s=15, alpha=0.6)

    # 设置标题和坐标轴标签
    plt.title("2D Projection (D1 - D2)", fontsize=12, fontweight='bold')
    plt.xlabel(dim_names[0], fontsize=10)
    plt.ylabel(dim_names[1], fontsize=10)

    # 网格
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 调用绘图
# file_path = '2D-safe-sample-10000.csv'
file_path ='safe_points_terminal_constraint_new_npoint=1.csv'
draw_2d_projection_fixed(file_path)
