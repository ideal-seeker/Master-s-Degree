import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import scipy.linalg as LA
import polytope as pc
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False
# 系统参数

# num_safe_need = 1
# num_unsafe_need = 1
def random_x0(x_min, x_max):
    x0 = np.random.uniform(x_min-0.1, x_max+0.1, (1, 2)).squeeze(0)
    return x0
def dare_lqr_gain(A, B, Q, R):
    """
    输入: A, B, Q, R
    输出: P (DARE解), K (LQR增益)
    """
    P = LA.solve_discrete_are(A, B, Q, R)
    K = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return P, K


def compute_terminal_set(A, B, K, x_min, x_max, u_min, u_max, max_iter=20):
    """
    使用多面体迭代法求二维终端不变集
    输出: polytope 对象 Xf
    """
    # 状态约束 Hx x <= hx
    Hx = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    hx = np.array([x_max, -x_min, x_max, -x_min])

    # 输入约束 Hu u <= hu -> Hu K x <= hu
    Hu = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    hu = np.array([u_max, -u_min, u_max, -u_min])

    # 初始集合: 状态+输入约束
    Xf = pc.Polytope(np.vstack([Hx, Hu @ K]), np.hstack([hx, hu]))

    for i in range(max_iter):
        # 计算预像: Pre(Xf) = {x | (A+BK)x in Xf}
        Xf_pre = pc.Polytope(Xf.A @ (A + B @ K), Xf.b)
        # 与原集合取交集
        Xf_new = Xf.intersect(Xf_pre)

        # 收敛判断（顶点数不变）
        if len(Xf_new.A) == len(Xf.A):
            break
        Xf = Xf_new

    return Xf
# MPC 求解器函数
def mpc_solver(x0, A, B, Q, R, P, Xf, N, x_min, x_max, u_min, u_max):
    x = cp.Variable((2, N + 1))
    u = cp.Variable((2, N))

    # 目标函数
    cost = 0
    for k in range(N):
        cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
    cost += cp.quad_form(x[:, N], P)

    # 约束
    constraints = [x[:, 0] == x0]

    for k in range(N):
        constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
        constraints += [u_min <= u[0, k], u[0, k] <= u_max]
        constraints += [u_min <= u[1, k], u[1, k] <= u_max]
        constraints += [x_min <= x[0, k], x[0, k] <= x_max]
        constraints += [x_min <= x[1, k], x[1, k] <= x_max]

    # 末端状态约束 x_N ∈ Xf (H x_N <= h)
    constraints += [Xf.A @ x[:, N] <= Xf.b]

    # 求解
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS, verbose=True)

    if prob.status != cp.OPTIMAL:
        print("MPC求解失败")
        return False, x.value, u.value
    return True, x.value, u.value


def plot_distribution_statistics_simultaneous(data, title_prefix="X和Y的分布统计"):
    """
    绘制x和y的分布统计图，所有图表同时显示

    参数:
    data: N行2列的数组，每行是一个二维向量 [x, y]
    title_prefix: 图表标题前缀
    """

    # 将数据分离为x和y
    x = data[:, 0]
    y = data[:, 1]

    # 创建多个figure，但不立即显示
    figures = []

    # 1. 散点图
    fig1 = plt.figure(figsize=(10, 8))
    plt.scatter(x, y, alpha=0.6, color='blue')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{title_prefix} - X-Y散点图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig1)

    # 2. X的直方图
    fig2 = plt.figure(figsize=(10, 6))
    plt.hist(x, bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('X')
    plt.ylabel('频数')
    plt.title(f'{title_prefix} - X的分布直方图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig2)

    # 3. Y的直方图
    fig3 = plt.figure(figsize=(10, 6))
    plt.hist(y, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Y')
    plt.ylabel('频数')
    plt.title(f'{title_prefix} - Y的分布直方图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig3)

    # 4. X的箱线图
    fig4 = plt.figure(figsize=(8, 6))
    plt.boxplot(x, vert=True)
    plt.ylabel('X值')
    plt.title(f'{title_prefix} - X的箱线图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig4)

    # 5. Y的箱线图
    fig5 = plt.figure(figsize=(8, 6))
    plt.boxplot(y, vert=True)
    plt.ylabel('Y值')
    plt.title(f'{title_prefix} - Y的箱线图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig5)

    # 6. 联合分布热图
    fig6 = plt.figure(figsize=(10, 8))
    h = plt.hist2d(x, y, bins=20, cmap='Blues')
    plt.colorbar(h[3])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{title_prefix} - X-Y联合分布热图')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig6)

    # 7. 并排箱线图比较
    fig7 = plt.figure(figsize=(8, 6))
    plt.boxplot([x, y], labels=['X', 'Y'])
    plt.ylabel('值')
    plt.title(f'{title_prefix} - X和Y的箱线图比较')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig7)

    # 显示所有图表
    plt.show()

    # 打印统计摘要
    print_statistics_summary(x, y)

    return figures


def print_statistics_summary(x, y):
    """打印统计摘要"""
    print("=" * 50)
    print("统计摘要")
    print("=" * 50)
    print(f"X的统计信息:")
    print(f"  样本数量: {len(x)}")
    print(f"  均值: {np.mean(x):.4f}")
    print(f"  标准差: {np.std(x):.4f}")
    print(f"  最小值: {np.min(x):.4f}")
    print(f"  最大值: {np.max(x):.4f}")
    print(f"  中位数: {np.median(x):.4f}")

    print(f"\nY的统计信息:")
    print(f"  样本数量: {len(y)}")
    print(f"  均值: {np.mean(y):.4f}")
    print(f"  标准差: {np.std(y):.4f}")
    print(f"  最小值: {np.min(y):.4f}")
    print(f"  最大值: {np.max(y):.4f}")
    print(f"  中位数: {np.median(y):.4f}")

    # 计算相关系数
    correlation = np.corrcoef(x, y)[0, 1]
    print(f"\nX和Y的相关系数: {correlation:.4f}")


if __name__ == "__main__":
    row_sample_safedata = 32
    row_sample_unsafedata = 32
    A = np.array([[2, 1], [-1, 2]], dtype=np.float32)
    B = np.array([[1, 0], [0, 1]], dtype=np.float32)
    Q = np.diag([1, 1])
    R = 100 * np.diag([1, 1])

    N = 10  # 预测时域
    u_min = -1  # 控制输入下界
    u_max = 1  # 控制输入上界
    x_min = -1  # 状态x下界
    x_max = 1  # 状态x上界
    n_stop = 1
    # num_safe_need = 10000
    # num_unsafe_need = 9000
    num_safe_need = 1000
    num_unsafe_need = 1


    # 1. 求P,K
    P, K = dare_lqr_gain(A, B, Q, R)
    print("DARE终端成本P:\n", P)
    print("LQR增益K:\n", K)

    # 2. 求终端不变集
    Xf = compute_terminal_set(A, B, K, x_min, x_max, u_min, u_max)
    print("终端不变集多面体 H:\n", Xf.A)
    print("终端不变集多面体 h:\n", Xf.b)

    # 生成300个初始点

    all_trajectories = []
    all_trajectories_unsafe = []
    all_umemories = []
    all_umemories_unsafe = []
    # 通过MPC控制每个初始点
    num_safe = 0
    num_unsafe = 0
    while num_safe < num_safe_need or num_unsafe < num_unsafe_need:
        x0 = random_x0(x_min, x_max)
        RES, x_trajectory, u_trajectory = mpc_solver(x0, A, B, Q, R, P, Xf, N, x_min, x_max, u_min, u_max)
        while RES is False:
            all_trajectories_unsafe.append(x0)
            num_unsafe += 1
            x0 = np.random.uniform(-2, 2, (1, 2)).squeeze(0)
            RES, x_trajectory, u_trajectory = mpc_solver(x0, A, B, Q, R, P, Xf, N, x_min, x_max, u_min, u_max)
        x_traj_short = x_trajectory[:, :n_stop]  # 取前5个点
        u_traj_short = u_trajectory[:, :n_stop]
        all_trajectories.append(x_traj_short .T)  # 记录轨迹，转置为 (n_stop, 2)
        num_safe += n_stop
        all_umemories.append( u_traj_short.T)
        print(f'num_safe:{num_safe}，num_unsafe：{num_unsafe}')
    # 如果没有成功的轨迹，打印消息
    if not all_trajectories:
        print("No trajectories were successfully generated.")
    #
    # 将轨迹数据转换为TensorDataset
    trajectories = np.array(all_trajectories)
    # x_tensor = torch.tensor(trajectories[:, :, 0], dtype=torch.float32)
    # y_tensor = torch.tensor(trajectories[:, :, 1], dtype=torch.float32)

    trajectories_unsafe = np.array(all_trajectories_unsafe)
    # x_tensor_unsafe = torch.tensor(trajectories_unsafe[:, 0], dtype=torch.float32)
    # y_tensor_unsafe = torch.tensor(trajectories_unsafe[:, 1], dtype=torch.float32)

    safe_points =trajectories.reshape(-1,2)
    unsafe_points = trajectories_unsafe.reshape(-1,2)

    df1 = pd.DataFrame(safe_points, columns=['X', 'Y'])
    df1.to_csv(f'safe_points_terminal_constraint_new_npoint={n_stop}.csv', index=False)

    # 将第二个数组保存为 CSV 文件
    df2 = pd.DataFrame(unsafe_points, columns=['X', 'Y'])
    df2.to_csv(f'unsafe_points_terminal_constraint_npoint={n_stop}.csv', index=False)

    print("两个数组已保存为 CSV 文件")
    x1, y1 = safe_points[:, 0], safe_points[:, 1]  # array1 的 x 和 y
    x2, y2 = unsafe_points[:, 0], unsafe_points[:, 1]  # array2 的 x 和 y

    plot_distribution_statistics_simultaneous(safe_points, title_prefix="X和Y的分布统计")
        # 绘制统计图WWA
