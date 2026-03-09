import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 从CSV文件读取数据
file_path = '9D-safe-sample-10000.csv'


def draw_9d_projections_fixed(file_path, plots_per_group=6):
    """
    绘制9D数据的二维投影图（修复标题显示问题）
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    try:
        df = pd.read_csv(file_path)
        if df.shape[1] > 9:
            df = df.iloc[:, :9]
        data_array = df.values
        print(f"成功读取数据: {data_array.shape}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 生成所有二维投影组合
    dim_combinations = []
    for i in range(9):
        for j in range(i + 1, 9):
            dim_combinations.append((i, j))

    num_groups = (len(dim_combinations) + plots_per_group - 1) // plots_per_group
    dimension_names = [f'D{i + 1}' for i in range(9)]

    print(f"9D数据共有 {len(dim_combinations)} 个二维投影")
    print(f"分为 {num_groups} 组显示")

    for group in range(num_groups):
        start_idx = group * plots_per_group
        end_idx = min((group + 1) * plots_per_group, len(dim_combinations))
        current_combinations = dim_combinations[start_idx:end_idx]

        num_plots = len(current_combinations)
        ncols = min(3, num_plots)
        nrows = (num_plots + ncols - 1) // ncols

        # 创建图形，增加高度给标题留足够空间
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))

        # 扁平化axes数组
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()

        for idx, (dim1, dim2) in enumerate(current_combinations):
            ax = axes[idx]
            x_data = data_array[:, dim1]
            y_data = data_array[:, dim2]

            ax.scatter(x_data, y_data, color='blue', s=15, alpha=0.6)
            ax.set_title(f'D{dim1 + 1}-D{dim2 + 1}', fontsize=11, fontweight='bold')
            ax.set_xlabel(dimension_names[dim1], fontsize=9)
            ax.set_ylabel(dimension_names[dim2], fontsize=9)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(num_plots, len(axes)):
            axes[idx].set_visible(False)

        # 先调整布局，再添加标题
        plt.tight_layout()

        # 添加总标题，位置更保守
        # plt.suptitle(f'Group {group + 1}/{num_groups}',
        #              fontsize=12, fontweight='bold', y=0.995)

        plt.show()


# 执行绘图
draw_9d_projections_fixed(file_path, plots_per_group=6)