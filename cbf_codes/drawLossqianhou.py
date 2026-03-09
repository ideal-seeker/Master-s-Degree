import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# -----------------------
# 中文字体设置（防乱码）
# -----------------------
rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# -----------------------
# 数据生成
# -----------------------
x = np.arange(1, 101)            # 样本索引 1:100
y = 0.5*np.random.randn(100)    # 以 1 为中心的随机采样
y_sgn = np.sign(y)               # sgn(h(x))

# -----------------------
# 绘图
# -----------------------
plt.figure(figsize=(8, 6))

# ===== 上子图：h(x) =====
plt.subplot(2, 1, 1)
plt.plot(
    x, y,
    color='blue',
    marker='o',
    linestyle='-',
    markersize=4,
    label=r'$h(x)$'
)

# h(x) = 0 红色参考线
plt.axhline(
    y=0,
    color='red',
    linestyle='--',
    linewidth=2,
    label=r'$h(x)=0$'
)

# h(x) = 1 紫色参考线（新增）
# plt.axhline(
#     y=1,
#     color='purple',
#     linestyle='-.',
#     linewidth=2,
#     label=r'$h(x)=1.2$'
# )

plt.xlabel('样本索引', fontsize=14)
plt.ylabel(r'$h(x)$', fontsize=14)
plt.title(r'$h(x)$', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)

# ===== 下子图：sgn(h(x)) =====
plt.subplot(2, 1, 2)
plt.plot(
    x, y_sgn,
    color='green',
    marker='o',
    linestyle='-',
    markersize=4,
    label=r'$\mathrm{sgn}(h(x))$'
)

# sgn(h(x)) = 0 红色参考线
plt.axhline(
    y=0,
    color='red',
    linestyle='--',
    linewidth=2,
    label=r'$\mathrm{sgn}(h(x))=0$'
)

plt.xlabel('样本索引', fontsize=14)
plt.ylabel(r'$\mathrm{sgn}(h(x))$', fontsize=14)
plt.title(r'$\mathrm{sgn}(h(x))$', fontsize=16)
plt.yticks([-1, 0, 1])
plt.grid(True)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()
