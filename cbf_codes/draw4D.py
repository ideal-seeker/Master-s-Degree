import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 从CSV文件读取数据
file_path = 'safe_points_terminal_constraints_nstop=1.csv'


def detailed_4d_analysis(file_path):
    """
    详细分析四维数据并绘制多个可视化图表
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    try:
        df = pd.read_csv(file_path)
        # 确保只有4列数据
        if df.shape[1] > 4:
            df = df.iloc[:, :4]
            print(f"使用前4列数据，忽略其他列")

        data_array = df.values
        print(f"成功读取数据: {data_array.shape}")

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 创建大图包含多个子图
    fig = plt.figure(figsize=(20, 12))
    # 六个二维投影子图：1-2, 1-3, 1-4, 2-3, 2-4, 3-4
    dim_combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    dimension_names = ['Dim 1', 'Dim 2', 'Dim 3', 'Dim 4']

    for i, (dim1, dim2) in enumerate(dim_combinations):
        ax = fig.add_subplot(2, 3, i + 1)  # 2行3列布局

        x_data = data_array[:, dim1]
        y_data = data_array[:, dim2]

        # 使用蓝色点绘制
        scatter = ax.scatter(x_data, y_data, color='blue', s=20, alpha=0.6)

        ax.set_title(f'{dimension_names[dim1]} vs {dimension_names[dim2]}', fontweight='bold')
        ax.set_xlabel(f'{dimension_names[dim1]}')
        ax.set_ylabel(f'{dimension_names[dim2]}')
        ax.grid(True, alpha=0.3)

        # 添加数据点数量信息
        ax.text(0.02, 0.98, f'n={len(data_array)}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.suptitle(f'2D Projections of 4D Safe Points\n{file_path}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.show()

    # 可选：单独显示每个投影图（如果需要更清晰的视图）
    print("\n生成单独的投影图...")
    fig_individual, axes_individual = plt.subplots(2, 3, figsize=(18, 10))
    axes_individual = axes_individual.flatten()

    for i, (dim1, dim2) in enumerate(dim_combinations):
        ax = axes_individual[i]
        x_data = data_array[:, dim1]
        y_data = data_array[:, dim2]

        ax.scatter(x_data, y_data, color='blue', s=30, alpha=0.7)
        ax.set_title(f'{dimension_names[dim1]} vs {dimension_names[dim2]}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{dimension_names[dim1]}', fontsize=12)
        ax.set_ylabel(f'{dimension_names[dim2]}', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return data_array


# 执行详细分析
data = detailed_4d_analysis(file_path)