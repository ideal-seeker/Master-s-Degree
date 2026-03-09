import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import scipy.linalg as LA
import polytope as pc

def random_x0(x_min, x_max):
    x_random = np.random.uniform(x_min*1.2, x_max*1.2, size=(1, len(x_min)))
    return x_random

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
    使用多面体迭代法求四维终端不变集
    输出: polytope 对象 Xf
    """
    n = len(x_min)
    m = 1

    # 状态约束 Hx x <= hx
    Hx = np.vstack([np.eye(n), -np.eye(n)])
    hx = np.hstack([x_max, -x_min])

    # 输入约束 Hu u <= hu -> Hu K x <= hu
    Hu = np.vstack([np.eye(m), -np.eye(m)])
    hu = np.hstack([u_max, -u_min])

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
    n = A.shape[0]
    m = B.shape[1]

    x = cp.Variable((n, N + 1))
    u = cp.Variable((m, N))

    # 目标函数
    cost = 0
    for k in range(N):
        cost += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
    cost += cp.quad_form(x[:, N], P)

    # 约束
    constraints = [x[:, 0] == x0]

    for k in range(N):
        constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
        for j in range(m):
            constraints += [u_min[j] <= u[j, k], u[j, k] <= u_max[j]]
        for j in range(n):
            constraints += [x_min[j] <= x[j, k], x[j, k] <= x_max[j]]

    # 末端状态约束 x_N ∈ Xf (H x_N <= h)
    constraints += [Xf.A @ x[:, N] <= Xf.b]

    # 求解
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS, verbose=True)

    if prob.status != cp.OPTIMAL:
        print("MPC求解失败")
        return False, x.value, u.value
    return True, x.value, u.value

def plot2x(i, j, safe_points, unsafe_points):
    plt.figure()
    x1, y1 = safe_points[:, i - 1], safe_points[:, j - 1]  # array1 的 x 和 y
    x2, y2 = unsafe_points[:, i - 1], unsafe_points[:, j - 1]  # array2 的 x 和 y

    # 绘制散点图
    plt.scatter(x1, y1, label='Dataset 1', color='blue', alpha=0.5)
    # plt.scatter(x2, y2, label='Dataset 2', color='red', alpha=0.5)

    # 添加标题和标签
    plt.title(f"x{i}-x{j}")
    plt.xlabel(f"x{i}")
    plt.ylabel(f"x{j}")
    plt.legend()
    plt.xlim(x_min[i - 1], x_max[i - 1])
    plt.ylim(x_min[j - 1], x_max[j - 1])

if __name__ == "__main__":
    #四维mpc求解
    dim_x = 4
    dim_u = 1
    # 系统参数
    row_sample_safedata = 32
    row_sample_unsafedata = 32
    A = np.array([[1, 0.1, 0, 0], [0, 0.9818, 0.2673, 0], [0, 0, 1, 0.1], [0, -0.0455, 3.1182, 1]], dtype=np.float32)
    B = np.array([[0], [0.1818], [0], [0.4546]], dtype=np.float32)
    Q = np.diag([2, 2, 2, 2])
    R = 1*np.diag([1])
    # P = 1*np.diag([1, 1, 1, 1])
    N = 30  # 预测时域
    u_min = np.array([-1])
    u_max = np.array([1])
    x_min = np.array([-1, -1.5, -0.35, -1])  # 状态x下界
    x_max = np.array([1, 1.5, 0.35, 1])  # 状态x上界
    n_stop = 1
    # num_safe_need = 10000
    # num_unsafe_need = 1
    num_safe_need = 1000
    num_unsafe_need = 1
    # 终端约束
    error = 5e-2


    all_trajectories = []
    all_trajectories_unsafe = []
    all_umemories = []
    all_umemories_unsafe = []
    # 通过MPC控制每个初始点
    num_safe = 0
    num_unsafe = 0

    time_start = time.time()
    time1 = time.time()
    # 1. 求P,K
    P, K = dare_lqr_gain(A, B, Q, R)
    print("DARE终端成本P:\n", P)
    print("LQR增益K:\n", K)

    # 2. 求终端不变集
    Xf = compute_terminal_set(A, B, K, x_min, x_max, u_min, u_max, max_iter=20)
    print("终端不变集多面体 H:\n", Xf.A)
    print("终端不变集多面体 h:\n", Xf.b)
    while num_safe < num_safe_need or num_unsafe < num_unsafe_need:
        x0 = random_x0(x_min, x_max).flatten()
        RES, x_trajectory, u_trajectory = mpc_solver(x0, A, B, Q, R, P, Xf, N, x_min, x_max, u_min, u_max)
        while RES is False:
            all_trajectories_unsafe.append(x0)
            num_unsafe += 1
            x0 = random_x0(x_min, x_max).flatten()
            RES, x_trajectory, u_trajectory = mpc_solver(x0, A, B, Q, R, P, Xf, N, x_min, x_max, u_min, u_max)
        x_traj_short = x_trajectory[:, :n_stop]  # 取前nstop个点
        u_traj_short = u_trajectory[:, :n_stop]
        all_trajectories.append(x_traj_short.T)  # 记录轨迹，转置为 (N+1, 2)
        num_safe += n_stop
        all_umemories.append(u_traj_short.T)
        time2 = time.time()

        print(f'num_safe:{num_safe}/{num_safe_need}，num_unsafe：{num_unsafe}/{num_unsafe_need}，距离上一次打印时间间隔{(time2-time1):.2f}s')
        time1 = time.time()
        # 如果没有成功的轨迹，打印消息
        time_end = time.time()
        time_all = time_end-time_start
        print(f'总用时{time_all}s')
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

    safe_points =trajectories.reshape(-1, dim_x)
    unsafe_points = trajectories_unsafe.reshape(-1, dim_x)

    dataset_safe = TensorDataset(torch.tensor(safe_points))  # 这里已经有轨迹的dataset了
    dataset_unsafe = TensorDataset(torch.tensor(unsafe_points))  # 这里已经有轨迹的dataset了

    safe_dataloader = DataLoader(dataset_safe, batch_size=row_sample_safedata, shuffle=True)
    unsafe_dataloader = DataLoader(dataset_unsafe, batch_size=row_sample_unsafedata, shuffle=True)


    df1 = pd.DataFrame(safe_points, columns=['X1', 'X2', 'X3', 'X4'])
    df1.to_csv('safe_points_terminal_constraints_nstop=1.csv', index=False)

    # 将第二个数组保存为 CSV 文件
    df2 = pd.DataFrame(unsafe_points, columns=['X1', 'X2', 'X3', 'X4'])
    df2.to_csv('unsafe_terminal_constraints_nstop=1.csv', index=False)

    print("两个数组已保存为 CSV 文件")
    # x1-x2
for i in range(1, 4+1):
    for j in range(i+1, 4+1):
        plot2x(i, j, safe_points, unsafe_points)
# 显示图形
plt.show()
print(0)