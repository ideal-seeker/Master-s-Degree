import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def combine_six_figures():
    """
    合成指定文件夹中的六张figure图片
    """
    # 指定文件夹路径
    folder_path = r'C:\Users\XJ_lab\Desktop\实验仿真结果图\10.11-9D修改代码结果9D\epsilon=0.075Polyhedron_h-network'

    # 构建六张图片的完整路径
    image_paths = [
        os.path.join(folder_path, 'figure_1.png'),
        os.path.join(folder_path, 'figure_2.png'),
        os.path.join(folder_path, 'figure_3.png'),
        os.path.join(folder_path, 'figure_4.png'),
        os.path.join(folder_path, 'figure_8.png'),
        os.path.join(folder_path, 'figure_8.png')
    ]

    # 检查所有图片是否存在
    missing_files = []
    for path in image_paths:
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        print("以下文件不存在:")
        for file in missing_files:
            print(f"  {file}")
        return

    print("所有图片文件都存在，开始合成...")

    # 自定义标题（根据你的维度投影）
    custom_titles = [
        'Dimension 1-2 Projection',
        'Dimension 1-3 Projection',
        'Dimension 1-4 Projection',
        'Dimension 2-3 Projection',
        'Dimension 2-4 Projection',
        'Dimension 3-4 Projection'
    ]

    # 创建2行3列的子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 加载并显示每张图片
    for i, (ax, img_path, title) in enumerate(zip(axes, image_paths, custom_titles)):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        # ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')  # 隐藏坐标轴

        # 添加子图标签 (a), (b), ...
        label = chr(97 + i)  # 97是'a'的ASCII码
        ax.text(0.02, 0.98, f'({label})', transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')

    # 添加总标题
    plt.suptitle('9D Training', fontsize=20, fontweight='bold', y=0.95)

    # 调整布局并保存
    plt.tight_layout()

    # 保存到同一文件夹
    output_path = os.path.join(folder_path, 'combined_six_figures.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"合成完成！图片已保存到: {output_path}")
    print(f"图片包含: {len(image_paths)} 个子图")


# 执行合成
combine_six_figures()