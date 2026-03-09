import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from flatbuffers.packer import float32
from datetime import date
from scipy.linalg import solve
import numpy as np
import cvxpy as cp
import scipy
import pandas as pd
import random
import time
import itertools
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice
from scipy.optimize import linprog
from PIL import Image
from scipy.spatial import ConvexHull
from itertools import combinations
import os
import cvxpylayers.torch as cvxpy_torch
from shapely.geometry import Polygon
from shapely.ops import unary_union

"""非常慢，时间为充分条件的上千倍"""
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False
# 此处为全局变量
# 系统参数

global train_times
global optimal_times
global compute_cbfloss_times
global res
train_times = []
optimal_times = []
kept_times = []
# compute_cbfloss_times = []
res = []
def set_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)  # 设置 CPU 随机种子
    torch.cuda.manual_seed(seed)  # 设置 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机种子（如果使用多个GPU）
    np.random.seed(seed)  # 设置 numpy 随机种子
    random.seed(seed)  # 设置 Python 随机模块的种子
    torch.backends.cudnn.deterministic = True  # 确保结果确定性
    torch.backends.cudnn.benchmark = False  # 在确定性计算中关闭性能优化


# 调用 set_seed 函数，设置固定的种子

def dataset_construct(safe_name, unsafe_name):
    # 数据集的范围是[-2,2]*[-2,2]
    # 导入安全数据):
    # 导入安全数据
    df1_loaded = pd.read_csv(safe_name)
    safe_points = df1_loaded.to_numpy()

    # 导入不安全数据
    df2_loaded = pd.read_csv(unsafe_name)
    unsafe_points = df2_loaded.to_numpy()

    # 统计安全和不安全数据数量
    safe_points_num = len(safe_points)
    unsafe_points_num = len(unsafe_points)

    # batch的大小，先设成是整个数据集更新一次
    batch_num_of_safe = safe_points_num // num_batches
    batch_num_of_unsafe = unsafe_points_num // num_batches

    # 构造数据集
    dataset_safe = TensorDataset(torch.tensor(safe_points, dtype=torch.float32))
    dataset_unsafe = TensorDataset(torch.tensor(unsafe_points, dtype=torch.float32))

    data_loader_safe = DataLoader(dataset_safe, batch_size=batch_num_of_safe, shuffle=True)
    data_loader_unsafe = DataLoader(dataset_unsafe, batch_size=batch_num_of_unsafe, shuffle=True)

    # 列表化数据集（后面可能用得到）
    all_batches_safe = list(data_loader_safe)
    all_batches_unsafe = list(data_loader_unsafe)

    return data_loader_safe, data_loader_unsafe, all_batches_safe, all_batches_unsafe, batch_num_of_safe, batch_num_of_unsafe, safe_points, unsafe_points
def create_9d_simplex_from_data(data, n_vertices=200, c=0.1):
    """
    简化版本：从9维数据生成正单形平面参数
    """
    start_time = time.time()
    # 计算半径
    center = np.zeros(9)
    radius = np.max(np.linalg.norm(data - center, axis=1))

    # 生成均匀球面点
    vertices = np.random.randn(n_vertices, 9)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius

    # 计算凸包平面
    hull = ConvexHull(vertices)
    w_vectors = hull.equations[:, :-1]
    b_values = hull.equations[:, -1]
    w_vectors_10d = []
    b_values_10d = []
    for i in range(w_vectors.shape[0]):
        # 从w_vectors提取w1-w9
        w1, w2, w3, w4, w5, w6, w7, w8, w9 = w_vectors[i]
        # 从b_values提取对应的b
        b = b_values[i]
        if abs(b) < 1e-10:
            continue  # 跳过数值不稳定的平面

        w1_10d = c * w1 / b
        w2_10d = c * w2 / b
        w3_10d = c * w3 / b
        w4_10d = c * w4 / b
        w5_10d= c * w5 / b
        w6_10d = c * w6 / b
        w7_10d= c * w7 / b
        w8_10d = c * w8 / b
        w9_10d = c * w9 / b

        b_10d = c

        w_vectors_10d.append([w1_10d, w2_10d, w3_10d, w4_10d, w5_10d, w6_10d, w7_10d, w8_10d, w9_10d])
        b_values_10d.append(b_10d)

    w_vectors_10d_tensor = torch.tensor(w_vectors_10d, dtype=torch.float32)
    b_values_10d_tensor = torch.tensor(b_values_10d, dtype=torch.float32)
    print(f"最终凸包边数: {n_vertices}")
    end_time = time.time()

    gouzaotime = end_time-start_time
    return w_vectors_10d_tensor, b_values_10d_tensor, gouzaotime

class CBFNetwork_Polygon(nn.Module):
    def __init__(self, state_dim, w, b, num_networks=13):
        super(CBFNetwork_Polygon, self).__init__()
        self.num_networks = num_networks
        self.fc_list = nn.ModuleList([nn.Linear(state_dim, 1) for _ in range(num_networks)])

        for i, fc in enumerate(self.fc_list):
            # 假设 w[i] 的形状是 [1, 2]，直接用它初始化 fc.weight
            w_value = w[i].view(1, -1)
             # 确保 w[i] 是一个形状为 [1, 2] 的张量
            fc.weight.data = w_value.detach()
            # 偏置初始化
            init.constant_(fc.bias, b[i].detach())  # 初始化偏置
            # 保证偏置是可以训练的
            fc.bias.requires_grad = True

    def forward(self, x):
        h_list = [fc(x) for fc in self.fc_list]  # 每个输出 shape [batch, 1]
        h_all = torch.cat(h_list, dim=1)  # [batch, num_networks]
        h_min = torch.min(h_all, dim=1)[0]  # [batch]
        return h_min, h_all

class CBFNetwork_Random(nn.Module):
    def __init__(self, state_dim, num_networks=13):
        super(CBFNetwork_Random, self).__init__()
        self.num_networks = num_networks
        self.fc_list = nn.ModuleList([nn.Linear(state_dim, 1) for _ in range(num_networks)])

        for fc in self.fc_list:
            init.constant_(fc.bias, 0.1)  # 偏置初始化
            fc.bias.requires_grad = True  # 偏置参与更新

    def forward(self, x):
        h_list = [fc(x) for fc in self.fc_list]  # 每个输出 shape [batch, 1]
        h_all = torch.cat(h_list, dim=1)  # [batch, num_networks]
        h_min = torch.min(h_all, dim=1)[0]  # [batch]
        return h_min, h_all

# ========== 神经网络参数提取，重组函数 ==========
def get_parameters(h_network):
    parameters = []
    for layer in h_network.fc_list:
        # 直接使用张量，不调用 detach()
        weight = layer.weight  # 保持为tensor以保留梯度
        bias = layer.bias  # 保持为tensor以保留梯度

        parameters.append({'weight': weight, 'bias': bias})

    return parameters


# 计算最优的u
def optimize_sup_u(P, cmax):
    m, n = P.shape
    # 使用cvxpylayers替代linprog以支持梯度传递

    u = cp.Variable(n)
    t = cp.Variable(m)

    P_param = cp.Parameter((m, n))
    cmax_param = cp.Parameter(m)

    residuals = cmax_param - P_param @ u
    constraints = [
        u[0] >= -59.3, u[0] <= 59.3,
        u[1] >= -59.3, u[1] <= 59.3,
        u[2] >= -59.3, u[2] <= 59.3,

        t >= 0,
        # T_max >= 0,
        t >= residuals+1e-3,      # 对不满足的约束,约束满足时会被t>=0覆盖，不满足时会有效
        # T_max >= t          # T_max 是最大的 t_i
]
    objective = cp.Minimize(cp.sum(t))
    prob = cp.Problem(objective, constraints)

    # layer = cvxpy_torch.CvxpyLayer(prob, parameters=[P_param, cmax_param], variables=[u, t, T_max])
    layer = cvxpy_torch.CvxpyLayer(prob, parameters=[P_param, cmax_param], variables=[u, t])
    # inv_cmax = 1.0 / (cmax + delta)
    u_star, _ = layer(P, cmax)
    J_star_result = torch.matmul(P, u_star.unsqueeze(-1))  # [m, 1]

    return J_star_result
def compute_P_and_c(params, B_tensor, A_tensor, x_safe_tensor, h_safe_min):
    P_list = []
    for param in params:
        w_i = param['weight']
        P_i = torch.matmul(w_i, B_tensor)
        P_list.append(P_i)
    # 堆叠P矩阵 - 保持Tensor
    P_tensor = torch.cat(P_list, dim=0)  # shape (num_network)
    batch_size = x_safe_tensor.shape[0]
    c_matrix = torch.zeros(batch_size, num_networks)

    for i, param in enumerate(params):
        w_i = param['weight']  # [1, 2]
        b_i = param['bias']  # [1]

        # 批量计算 w_i A x
        wA_i = torch.matmul(w_i, A_tensor)  # [1, 2]
        wAx = torch.matmul(x_safe_tensor, wA_i.t()).squeeze()  # [batch_size]

        # 批量计算 c_i
        c_i = (1 - alpha) * h_safe_min - b_i.squeeze() - wAx  # [batch_size]
        c_matrix[:, i] = c_i
        #这里改成找各维度最大的c命名为c_max
        c_max = c_matrix.max(dim=0)[0]
        # print(f"max_vector 是否需要梯度: {c_max.requires_grad}")
    return P_tensor, c_max, c_matrix.T
# def remove_dominated_columns(V, xsafe, params, num_per_piece=1, random_seed=None):#每个分片都有
#     """
#     改进版本（无需单独传 c_matrix）：
#     1. 保留每个维度最大值点
#     2. 每个分片优先选 c 最大的样本点，每个分片可选 num_per_piece 个
#        如果活跃样本不足 num_per_piece，保留全部活跃样本
#     3. 不再随机采样其他内点
#     输入:
#         V: torch.Tensor, shape [num_networks, num_samples]，每列对应一个样本的c向量
#         xsafe: torch.Tensor, shape [num_samples, state_dim]，对应样本x
#         params: list，包含每个网络的分片 affine 权重和bias
#         num_per_piece: int，每个分片最多选几个样本点
#     输出:
#         V_kept: torch.Tensor, shape [num_networks, n_kept]
#     """
#     M, N = V.shape
#     if random_seed is not None:
#         torch.manual_seed(random_seed)
#
#     # ---------- 1. 保留每个维度最大值点 ----------
#     dim_max_indices = []
#     for d in range(M):
#         max_val = torch.max(V[d, :])
#         max_mask = torch.abs(V[d, :] - max_val) < 1e-8
#         max_points = torch.where(max_mask)[0]
#         dim_max_indices.extend(max_points.tolist())
#     dim_max_idx = torch.unique(torch.tensor(dim_max_indices, device=V.device))
#
#     # ---------- 2. 每个分片优先选 c 最大点 ----------
#     piece_indices = []
#     num_networks = len(params)
#     for net_idx, param in enumerate(params):
#         w_i = param['weight']   # [num_pieces, input_dim] 或 [1, input_dim] 单分片时
#         b_i = param['bias']     # [num_pieces] 或 [1]
#
#
#         h_matrix = xsafe @ w_i.T + b_i  # [num_samples, num_pieces]
#
#
#         num_pieces = h_matrix.shape[1]
#         for piece_idx_i in range(num_pieces):
#             # 找出该分片活跃的点
#             active_mask = (h_matrix[:, piece_idx_i] >= h_matrix.max(dim=1)[0] - 1e-8)
#             active_indices = torch.where(active_mask)[0]
#
#             if len(active_indices) > 0:
#                 # 优先选 c 最大的 num_per_piece 个
#                 c_values = V[net_idx, active_indices]  # 每列是一个样本
#                 sorted_idx = torch.argsort(c_values, descending=True)
#                 n_select = min(num_per_piece, len(active_indices))
#                 chosen = active_indices[sorted_idx[:n_select]]
#                 piece_indices.extend(chosen.tolist())
#
#     piece_idx = torch.unique(torch.tensor(piece_indices, device=V.device))
#
#     # ---------- 3. 合并维度最大点和分片代表点 ----------
#     keep_idx = torch.cat([dim_max_idx, piece_idx])
#     keep_idx = torch.unique(keep_idx)
#     V_kept = V[:, keep_idx]
#
#     return V_kept
def remove_dominated_columns(V, max_inner_points=None, random_seed=None):
    """
    严格版本：只保留维度最大值点和随机内点
    """
    M, N = V.shape

    if random_seed is not None:
        torch.manual_seed(random_seed)

    # ---------- 1. 按第一维降序排序 ----------
    _, idx_sorted = torch.sort(V[0, :], descending=True)
    V_sorted = V[:, idx_sorted]

    # ---------- 2. 线性扫描，区分边界点和内点 ----------
    boundary_indices = []
    inner_indices = []
    current_max = V_sorted[:, 0].clone()
    boundary_indices.append(0)

    for i in range(1, N):
        v = V_sorted[:, i]
        if torch.all(current_max > v):
            inner_indices.append(i)
            continue
        boundary_indices.append(i)
        current_max = torch.maximum(current_max, v)

    # 映射回原始索引
    boundary_idx = idx_sorted[boundary_indices]
    inner_idx = idx_sorted[inner_indices]

    # ---------- 3. 从边界点中提取维度最大值点 ----------
    # 每个维度的最大值点
    dim_max_indices = []
    for d in range(M):
        # 在整个V中找该维度的最大值
        max_val = torch.max(V[d, :])
        # 找到所有达到最大值的点
        max_mask = torch.abs(V[d, :] - max_val) < 1e-8
        max_points = torch.where(max_mask)[0]
        dim_max_indices.extend(max_points.tolist())

    # 去重
    dim_max_idx = torch.unique(torch.tensor(dim_max_indices, device=V.device))

    # ---------- 4. 随机采样内点 ----------
    if max_inner_points is None:
        # 如果没有指定，保留所有内点
        sampled_inner = inner_idx
    else:
        # 随机采样指定数量的内点
        if len(inner_idx) > 0:
            n_sample = min(max_inner_points, len(inner_idx))
            perm = torch.randperm(len(inner_idx), device=V.device)
            sampled_inner = inner_idx[perm[:n_sample]]
        else:
            sampled_inner = torch.tensor([], device=V.device, dtype=torch.long)

    # ---------- 5. 合并：维度最大值点 + 随机内点 ----------
    keep_idx = torch.cat([dim_max_idx, sampled_inner])
    keep_idx = torch.unique(keep_idx)

    # ---------- 6. 返回结果 ----------
    V_kept = V[:, keep_idx]

    return V_kept
def compute_loss_N(h_network, x_safe, x_unsafe):
    """修改后的损失函数：增加 softmin 激活 loss"""
    A_tensor = torch.tensor(A, dtype=torch.float32)
    B_tensor = torch.tensor(B, dtype=torch.float32)
    # 安全 & 不安全数量统计
    N_safe, hardcount_safe, safepoints = safe_data_count(x_safe, h_network)
    N_unsafe, hardcount_unsafe = unsafe_data_count(x_unsafe, h_network)

    # 输入数据准备
    x_safe_tensor = torch.tensor(x_safe, dtype=torch.float32, requires_grad=False)
    x_unsafe_tensor = torch.tensor(x_unsafe, dtype=torch.float32, requires_grad=False)
    x_all_tensor = torch.cat((x_safe_tensor, x_unsafe_tensor), dim=0)

    # 网络前向传播
    h_safe_min, _ = h_network(x_safe_tensor)        # [B1]
    h_unsafe_min, _ = h_network(x_unsafe_tensor)    # [B2]
    h_all_min, h_all = h_network(x_all_tensor)      # [B1+B2], [B1+B2, num_networks]

    params = get_parameters(h_network)
    global epsilon
    epsilon = 0.075# 作用是拉大惩罚的范围！！！拉紧约束，错开阶跃点
    # Loss设计

    # 安全集合 loss
    loss_safe = torch.mean(torch.relu(-h_safe_min + epsilon))#从(-∞,0)被惩罚 到 (-∞,epsilon)被惩罚，h_safe_min收敛到epsilon

    # 不安全集合 loss
    loss_unsafe = torch.mean(torch.relu(h_unsafe_min + epsilon))#从(0,∞)被惩罚 到 (-epsilon,∞)被惩罚,h_unsafe_min收敛到-epsilon

    # CBF 条件 loss
    delta = 1e-6
    beta = 5.0
    J_res = []
    lss = []
    loss_cbf = 0
    CBF_satisfied_coordinates = []
    # 这里加一个函数得到P和c和J_star
    start = time.time()
    P, cmax, c = compute_P_and_c(params, B_tensor, A_tensor, x_safe_tensor, h_safe_min)
    # c_kept = remove_dominated_columns(
    #     V=c,
    #     xsafe=x_safe_tensor,
    #     params=params,
    #     num_per_piece=400//num_networks,  #
    #     random_seed=42
    # )
    c_kept = remove_dominated_columns(c, 200, 42)
    end = time.time()
    kept_times.append(end-start)
    start1 = time.time()
    _, kept_n = c_kept.shape
    print(f'dominate操作后剩余点数：{kept_n}')
    J_star_list = []
    for r in range(kept_n):
        c_here = c_kept[:, r]  # [num_networks]
        J_star = optimize_sup_u(P, c_here)  # [num_networks]
        J_star_list.append(J_star.unsqueeze(1))  # [num_networks,1]

    # [num_networks, kept_n]
    J_star_matrix = torch.cat(J_star_list, dim=1).squeeze(-1)  # [num_networks, kept_n, 1]

    # 1. 扩展维度以广播
    c_exp = c.unsqueeze(2)  # [20, 10000, 1]
    J_exp = J_star_matrix.unsqueeze(1)  # [20, 1, 157]

    # 2. 每个样本每个 J* 的 20 个分片维度是否都满足 c < J*
    satisfied_all_dims = (c_exp < J_exp).all(dim=0)  # [10000, 157]#某个ci满足所有维度都小于Jj吗

    # 3. 每个样本是否存在一个 J* 使 c 的 20 个维度都 < J* 的 20 个维度
    exists_J = satisfied_all_dims.any(dim=1)  # [10000]

    # 4. 惩罚 mask：没有任何 J* 满足 => 惩罚
    unsatisfied_mask = ~exists_J  # [10000]取反，不存在的被惩罚

    # 5. 取出需要惩罚的 residuals
    residuals_to_penalize = c_exp[:, unsatisfied_mask, :] - J_exp[:, :, :]  # [20, 惩罚样本数, 157]

    # 6. clamp 防止梯度爆炸
    residuals_to_penalize = torch.clamp(residuals_to_penalize, min=-5.0, max=5.0)

    # 7. softplus 惩罚并取平均
    if residuals_to_penalize.numel() > 0:
        loss_cbf = torch.nn.functional.softplus(residuals_to_penalize + epsilon / 5, beta=beta).mean()
    else:
        loss_cbf = torch.tensor(0.0, device=c.device)
    end1 = time.time()
    optimal_times.append(end1 - start1)
    print(f'本次求解所有优化问题用时{end1 - start1}')
    print(f"CBF loss: {loss_cbf.item()}")
    # CBF 满足的点统计
    with torch.no_grad():
        # exists_J: [num_samples], True 表示该样本满足 CBF 条件
        final_rows_positive = exists_J  # [10000]

        # 总满足点数
        total_valid_rows = final_rows_positive.sum().item()

        # 提取满足条件的坐标
        cbf_satisfied_points = x_safe_tensor[final_rows_positive].cpu().numpy()

        print(f"最终（并集）满足 CBF 条件的点数量: {total_valid_rows}/{x_safe_tensor.shape[0]}")

        # 保存结果
        CBF_result.append(total_valid_rows)
        CBF_satisfied_coordinates.append(cbf_satisfied_points)

        # safe 和 cbf 的并集交集分析
        cbf_set = set(tuple(p.tolist()) for p in cbf_satisfied_points)
        safe_set = set(tuple(point.tolist()) for point in safepoints)

        common_points = safe_set & cbf_set
        common_count = len(common_points)

        print(f"safepoints 中有 {common_count} 个元素在 CBF_satisfied_coordinates 中")
        print(f"占总 safepoints 的比例: {common_count / (len(safepoints) + 1):.2%}")

    def activation_entropy_loss(h_all, tau):
        softmin = torch.nn.functional.softmax(-h_all * tau, dim=1)  # [B, N]
        activation = softmin.sum(dim=0)
        activation_mean = softmin.mean(dim=0)# [N]
        entropy = -torch.sum(activation_mean * torch.log(activation_mean + 1e-8), dim=0)  # [B], 每行的熵
        print(f'熵：{-entropy}')
        return -entropy, activation_mean.detach()  # 负熵，训练时最大化熵

    act_loss, activation_counts = activation_entropy_loss(h_all, 2)

    #
    safe_penalty_here = 1
    unsafe_penalty_here = 20
    # safe_penalty_here = 0
    # unsafe_penalty_here = 0
    #1e-2挺好的
    cbf_penalty_here = 1e-2
    act_penalty_here = 0

    # 总 loss
    loss = (safe_penalty_here * loss_safe +
            unsafe_penalty_here * loss_unsafe +
            cbf_penalty_here * loss_cbf +
            act_penalty_here * act_loss)

    return loss, loss_safe, loss_unsafe, loss_cbf, total_valid_rows / x_safe_tensor.shape[0], hardcount_safe, hardcount_unsafe, activation_counts, act_loss, CBF_satisfied_coordinates, common_count, kept_n
# 网络训练函数
def safe_data_count(x_tensor, h_network):
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.unsqueeze(0)  # 单个样本转为[1,2]

    # 计算每个点的h值
    h_values, _ = h_network(x_tensor)

    # 方法1：使用sigmoid平滑近似来计算h(x)>0的条件
    smoothing_factor = 1000.0  # 控制平滑程度，越大越接近硬阈值
    condition_soft = torch.sigmoid(smoothing_factor * h_values)

    # 计算"软"计数（可微分）
    result = torch.sum(condition_soft)

    # 计算实际满足条件的点数（仅用于显示，不参与梯度传播）
    with torch.no_grad():
        hard_condition = h_values >= 0  # 布尔向量
        safe_points = x_tensor[hard_condition.squeeze()]  # 提取满足条件的点
        hard_count = torch.sum(hard_condition).item()
        safe_ratio = hard_count / x_tensor.shape[0]

    print(f"满足h(x)>0的点数: {hard_count}/{x_tensor.shape[0]} (软计数: {result.item():.2f})")

    return result, safe_ratio, safe_points


def unsafe_data_count(x_tensor, h_network):
    # 确保x_tensor的维度正确
    if len(x_tensor.shape) == 1:
        x_tensor = x_tensor.unsqueeze(0)  # 单个样本转为[1,2]

    # 计算每个点的h值
    h_values,_ = h_network(x_tensor)

    # 使用sigmoid平滑近似来计算h(x)<0的条件
    # 注意：这里使用-h_values，因为h<0等价于-h>0
    smoothing_factor = 1000.0  # 控制平滑程度，越大越接近硬阈值
    condition_soft = torch.sigmoid(smoothing_factor * (-h_values))

    # 计算"软"计数（可微分）
    result = torch.sum(condition_soft)

    # 计算实际满足条件的点数（仅用于显示，不参与梯度传播）
    with torch.no_grad():
        hard_count = torch.sum(h_values < 0).item()

    print(f"满足h(x)<0的点数: {hard_count}/{x_tensor.shape[0]} (软计数: {result.item():.2f})")

    return result, hard_count/x_tensor.shape[0]


def train_cbf_N(h_network, optimizer, data_loader_safe, data_loader_unsafe, batch_num_of_safe, batch_num_of_unsafe):
    loss_history = []
    loss_history_safe = []
    loss_history_unsafe = []
    loss_history_cbf = []
    loss_history_active = []
    TOTAL_VALID_ROWS = []
    HARDCOUNT_SAFE = []
    HARDCOUNT_UNSAFE = []
    Commom_counts = []
    kept_nall = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        total_loss_safe = 0
        total_loss_unsafe = 0
        total_loss_cbf = 0
        total_loss_active = 0
        count = 0
        for (x_safe,), (x_unsafe,) in zip(islice(data_loader_safe, batch_num_of_safe),
                                          islice(data_loader_unsafe, batch_num_of_unsafe)):
            count += 1
            optimizer.zero_grad()

            # 使用预计算的状态和CBF值计算损失
            (loss, loss_safe, loss_unsafe, loss_cbf, total_valid_rows, hardcount_safe, hardcount_unsafe, activation_counts, loss_active, CBF_satisfied_coordinates,
             common_count, kept_n) = compute_loss_N(h_network, x_safe, x_unsafe)

            loss.backward()
            optimizer.step()

            # 记录loss
            kept_nall+=kept_n
            total_loss += loss.item()
            total_loss_safe += loss_safe.item()
            total_loss_unsafe += loss_unsafe.item()
            total_loss_cbf += loss_cbf.item()
            total_loss_active += loss_active.item()

        avg_loss = total_loss / count
        avg_loss_safe = total_loss_safe / count
        avg_loss_unsafe = total_loss_unsafe / count
        avg_loss_cbf = total_loss_cbf / count
        avg_loss_active = total_loss_active / count


        loss_history.append(avg_loss)
        loss_history_safe.append(avg_loss_safe)
        loss_history_unsafe.append(avg_loss_unsafe)
        loss_history_cbf.append(avg_loss_cbf)
        loss_history_active.append(avg_loss_active)
        TOTAL_VALID_ROWS.append(total_valid_rows)
        HARDCOUNT_SAFE.append(hardcount_safe)
        HARDCOUNT_UNSAFE.append(hardcount_unsafe)
        Commom_counts.append(common_count)
        end_time = time.time()
        time_train = end_time - start_time
        train_times.append(time_train)
        num_active = (activation_counts > 0.045).sum().item()
        # print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}, Loss_safe: {avg_loss_safe:.6f}, Loss_unsafe: {avg_loss_unsafe:.6f}, Loss_cbf: {avg_loss_cbf:.6f}, Loss_active: {avg_loss_active:.6f},time:{time_train:.3f}s, 分片激活数（>5%）：{num_active}/{h_network.num_networks}\n")
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}, Loss_safe: {avg_loss_safe:.6f}, Loss_unsafe: {avg_loss_unsafe:.6f}, Loss_cbf: {avg_loss_cbf:.6f}, Loss_active: {avg_loss_active:.6f},time:{time_train:.3f}s\n")
        res.append(activation_counts)

    return loss_history, loss_history_safe, loss_history_unsafe, loss_history_cbf, loss_history_active, TOTAL_VALID_ROWS, HARDCOUNT_SAFE, HARDCOUNT_UNSAFE, CBF_satisfied_coordinates, Commom_counts, kept_nall

def plot_CBF_results(CBF_result,num_Xsafe):
    plt.figure()
    x = range(len(CBF_result))
    plt.plot(x, [x/num_Xsafe for x in CBF_result])
    plt.title('集合 $X_{safe}$ 中满足 CBF条件 的元素比例')
    plt.xlabel('训练次数')
    plt.ylabel('比例/100%')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # 数据是0-1之间的话用1.0
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()



def PlotDataGenerate(h_network):
    #点太多了画着慢
    n_dim = 5
    x1_grid = torch.linspace(x_min[0], x_max[0], n_dim)
    x2_grid = torch.linspace(x_min[1], x_max[1], n_dim)
    x3_grid = torch.linspace(x_min[2], x_max[2], n_dim)
    x4_grid = torch.linspace(x_min[3], x_max[3], n_dim)
    x5_grid = torch.linspace(x_min[4], x_max[4], n_dim)
    x6_grid = torch.linspace(x_min[5], x_max[5], n_dim)
    x7_grid = torch.linspace(x_min[6], x_max[6], n_dim)
    x8_grid = torch.linspace(x_min[7], x_max[7], n_dim)
    x9_grid = torch.linspace(x_min[8], x_max[8], n_dim)


    grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6, grid_x7, grid_x8, grid_x9 = torch.meshgrid(x1_grid, x2_grid, x3_grid, x4_grid, x5_grid, x6_grid, x7_grid, x8_grid, x9_grid, indexing='ij')
    grid_points = torch.cat([grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1),  grid_x3.reshape(-1, 1), grid_x4.reshape(-1, 1), grid_x5.reshape(-1, 1), grid_x6.reshape(-1, 1),  grid_x7.reshape(-1, 1), grid_x8.reshape(-1, 1), grid_x9.reshape(-1, 1)], dim=1)

    # 计算 h(x) 值（假设 h_network 已定义并训练）
    h_values, _ = h_network(grid_points)
    # h_values_reshaped = h_values.detach().numpy().reshape(100, 100, 35, 100)

    return grid_points, h_values, grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6, grid_x7, grid_x8, grid_x9
def plot2Dnew(h_network, i, j):
    """
    从分片线性网络 h(x)=min_i(w_i·x + b_i) 解析画出 h(x)=0 的多面体边界，
    并自动合并所有 h_i(x)>=0 的区域。
    不使用采样点、不使用凸包。
    """
    # 取出每个线性层的参数
    ws = []
    bs = []
    for fc in h_network.fc_list:
        w_i = fc.weight.detach().cpu().numpy().reshape(-1)  # (4,)
        b_i = fc.bias.detach().cpu().numpy().item()
        ws.append(w_i[[i-1, j-1]])
        bs.append(b_i)
    bs = np.array(bs)

    m = len(ws)

    xmin, xmax = x_min[i - 1], x_max[i - 1]
    ymin, ymax = x_min[j-1], x_max[j - 1]

    bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    all_polys = []  # 收集所有 h_i(x)>=0 的多边形区域

    for idx in range(m):
        w_i, b_i = ws[idx], bs[idx]

        # Step 1️⃣ 计算当前分片 i 的活跃区域
        A = []
        c = []
        for jdx in range(m):
            if idx == jdx:
                continue
            A.append(w_i - ws[jdx])
            c.append(bs[jdx] - b_i)
        A = np.array(A)
        c = np.array(c)

        # 用多边形裁剪出该分片活跃区域
        poly = np.array(bbox)
        for k in range(len(A)):
            a, rhs = A[k], c[k]
            new_poly = []
            for p1, p2 in zip(poly, np.roll(poly, -1, axis=0)):
                v1 = a @ p1 - rhs
                v2 = a @ p2 - rhs
                if v1 <= 0:  # p1 inside
                    new_poly.append(p1)
                if v1 * v2 < 0:  # 边界相交
                    t = v1 / (v1 - v2)
                    new_poly.append(p1 + t * (p2 - p1))
            poly = np.array(new_poly)
            if len(poly) == 0:
                break
        if len(poly) == 0:
            continue

        # Step 2️⃣ 再裁剪出 h_i(x) >= 0 的部分
        a = w_i
        new_poly = []
        for p1, p2 in zip(poly, np.roll(poly, -1, axis=0)):
            v1 = a @ p1 + b_i
            v2 = a @ p2 + b_i
            if v1 >= 0:
                new_poly.append(p1)
            if v1 * v2 < 0:  # 线段跨过0
                t = v1 / (v1 - v2)
                new_poly.append(p1 + t * (p2 - p1))
        poly_pos = np.array(new_poly)
        if len(poly_pos) < 3:
            continue

        all_polys.append(Polygon(poly_pos))

        # Step 3️⃣ 画出当前分片的 h_i(x)=0 边界线
        intersections = []
        for p1, p2 in zip(poly, np.roll(poly, -1, axis=0)):
            v1 = w_i @ p1 + b_i
            v2 = w_i @ p2 + b_i
            if v1 * v2 < 0:
                t = v1 / (v1 - v2)
                intersections.append(p1 + t * (p2 - p1))
            elif abs(v1) < 1e-8:
                intersections.append(p1)
        if len(intersections) >= 2:
            inters = np.array(intersections)
            plt.plot(inters[:, 0], inters[:, 1], color='black', linewidth=1.5)

    # Step 4️⃣ 合并所有分片区域并统一填色，label只出现一次
    if len(all_polys) > 0:
        union_poly = unary_union(all_polys)
        label_added = False  # 确保图例只加一次

        if union_poly.geom_type == 'Polygon':
            x, y = union_poly.exterior.xy
            plt.fill(x, y, color='#FFD700', alpha=0.618, label='$h(x)>=0$')
        elif union_poly.geom_type == 'MultiPolygon':
            for geom in union_poly.geoms:
                x, y = geom.exterior.xy
                if not label_added:
                    plt.fill(x, y, color='#FFD700', alpha=0.618, label='$h(x)>=0$')
                    label_added = True
                else:
                    plt.fill(x, y, color='#FFD700', alpha=0.618)

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('CIS Region')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

# 画凸包


# 画凸包
def plot2D2(grid_points, h_values):
    # 过滤满足 h(x) >= 0 条件的点
    mask = (h_values.detach().numpy() >= 0)
    points_in_range = grid_points[mask.flatten()]

    # 如果符合条件的点数量大于或等于 3，计算并绘制凸包
    if len(points_in_range) >= 3:
        hull = ConvexHull(points_in_range)

        # 填充凸包内部区域，设置透明度
        plt.fill(points_in_range[hull.vertices, 0], points_in_range[hull.vertices, 1], 'green', alpha=0.3,
                 label="PWLNN_CIS")

        # 绘制凸包的边界线
        for simplex in hull.simplices:
            plt.plot(points_in_range[simplex, 0], points_in_range[simplex, 1], color='#90EE90')

    # 设置图表标题和标签
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('CIS for 2D system')

    # 显示图例
    plt.legend()
    plt.grid(True)


# 画图 3D
def plot3D(grid_x, grid_y, h_values_reshaped):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 CBF 网络输出的三维曲面
    ax.plot_surface(grid_x.numpy(), grid_y.numpy(), h_values_reshaped, cmap='RdYlBu', alpha=0.8)

    # 绘制 h(x) = 1 的水平平面
    h_level = 0
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, h_level)
    ax.plot_surface(X, Y, Z, color='green', alpha=0.5, label='h(x) = 1')

    # 轴标签
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('h(x)')
    ax.set_title('CBF Network Output in 3D')

def plot_dataset(safe_points, unsafe_points):
    plt.figure(figsize=(15, 10))
    # x1, y1 = safe_points[:, 0], safe_points[:, 1]  # array1 的 x 和 y
    x2, y2 = unsafe_points[:, 0], unsafe_points[:, 1]  # array2 的 x 和 y

    # 绘制散点图
    # plt.scatter(x1, y1, label='Dataset 1', color='blue', alpha=0.5)
    plt.scatter(x2, y2, label='Dataset 2', color='red', alpha=0.5)

    # 添加标题和标签
    plt.title("Scatter Plot of Two 2D Arrays")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-2, 2)  # 设置x轴范围为 [-2, 2]
    plt.ylim(-2, 2)  # 设置y轴范围为 [-2, 2]
    plt.legend()
def plotloss(loss_history, loss_history_safe, loss_history_unsafe, loss_history_cbf, loss_history_active):
    plt.figure()
    x = range(len(loss_history_unsafe))
    plt.plot(x, loss_history_safe)
    plt.title('条件 1 对应的损失函数：$\mathrm{Loss}_{safe}$')
    plt.xlabel('训练次数')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.figure()
    plt.plot(x, loss_history_unsafe)
    plt.title('条件 2 对应的损失函数：$\mathrm{Loss}_{unsafe}$')
    plt.xlabel('训练次数')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.figure()
    plt.plot(x, loss_history_cbf)
    plt.title('条件 3 对应的损失函数：$\mathrm{Loss}_{CBF}$')
    plt.xlabel('训练次数')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.figure()
    plt.plot(x,loss_history)
    plt.title('总损失函数,$\mathrm{Loss}_{all}$')
    plt.xlabel('训练次数')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.figure()
    plt.plot(x,loss_history_active)
    plt.title('激活分片损失函数,$\mathrm{Loss}_{active}$')
    plt.xlabel('训练次数')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

def plot_CIS(points, axis1, axis2):
    # 选择两个轴的数据


    points_projected = points[:, [axis1 - 1, axis2 - 1]]

    if points_projected.shape[0] >= 3:  # 确保有足够的点来计算凸包
        hull = ConvexHull(points_projected)

        # 填充凸包内部并设置透明度
        plt.fill(points_projected[hull.vertices, 0], points_projected[hull.vertices, 1], '#ADD8E6', alpha=0.3,
                 label="Real_CIS")

        # 绘制凸包的边界
        for simplex in hull.simplices:
            plt.plot(points_projected[simplex, 0], points_projected[simplex, 1], color='#00008B')

    plt.grid(True)
    plt.legend(loc='upper right')

def plot_CISpoints(data, i, j):
    points_array = data[0]

    # # 如果符合条件的点数量大于或等于 3，计算并绘制凸包
    # if len(points_array) >= 3:
    #     hull = ConvexHull(points_array[:, [i-1, j-1]])
    #
    #     # 填充凸包内部区域，设置透明度
    #     plt.fill(points_array[hull.vertices, i-1], points_array[hull.vertices, j-1], 'blue', alpha=0.3,
    #              label="match CBF condition")
    #
    #     # 绘制凸包的边界线
    #     for simplex in hull.simplices:
    #         plt.plot(points_array[simplex, i-1], points_array[simplex, j-1], color='#4169E1')

    plt.scatter(points_array[:, i-1], points_array[:, j-1], color='#00008B', alpha=0.7, label="match CBF condition")

def plot_susnums(HARDCOUNT_SAFE, HARDCOUNT_UNSAFE):
    plt.figure()
    x = range(len(HARDCOUNT_UNSAFE))

    plt.plot(x, HARDCOUNT_SAFE, label='Safe: h(x) >= 0')

    # 可选：取消注释绘制不安全数据
    # plt.plot(x, HARDCOUNT_UNSAFE, label='Unsafe: h(x) < 0')
    plt.title('集合 $X_{safe}$ 中满足 $h(x) >= 0$ 的元素比例')
    plt.xlabel('训练次数')
    plt.ylabel('比例/100%')


    plt.figure()
    # 可选：取消注释绘制不安全数据
    plt.plot(x, HARDCOUNT_UNSAFE, label='Unsafe: h(x) < 0')

    plt.title('集合 $X_{unsafe}$ 中满足 $h(x) < 0$ 的元素比例')
    plt.xlabel('训练次数')
    plt.ylabel('比例/100%')

    # 设置纵坐标为百分比格式
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # 数据是0-1之间的话用1.0

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

def plot_safe_points(data, i, j):
    x = data[:, i - 1]
    y = data[:, j - 1]
    plt.scatter(x, y, alpha=0.6, s=30, label='safe points')

def plot_result(h_network, safe_points, unsafe_points,HARDCOUNT_SAFE, HARDCOUNT_UNSAFE,num_Xsafe,loss_history_safe, loss_history_unsafe, loss_history_cbf, loss_history_active, CBF_satisfied_coordinates ):
    grid_points, h_values, grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6, grid_x7, grid_x8, grid_x9 = PlotDataGenerate(h_network)

    plotloss(loss_history, loss_history_safe, loss_history_unsafe, loss_history_cbf, loss_history_active)
    plot_susnums(HARDCOUNT_SAFE, HARDCOUNT_UNSAFE)
    # CBF条件满足情况
    plot_CBF_results(CBF_result, num_Xsafe)
    # 画主图（2D）
    # 创建包含6个子图的图


    # plot_CIS(CIS_verticles, 1, 2)
    count = 0
    minicount = 0
    for m in range(1, 10):
        for n in range(m+1, 10):
            count += 1
            minicount += 1
            if count%6 == 1:
                minicount = 1
                plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, minicount)
            plot_CISpoints(CBF_satisfied_coordinates, m, n)
            plot_safe_points(safe_points, m, n)
            # plot2D2(grid_points, h_values, m, n)
            plot2Dnew(h_network, m, n)


def plot_result(h_network, safe_points, unsafe_points,HARDCOUNT_SAFE, HARDCOUNT_UNSAFE,num_Xsafe,loss_history_safe, loss_history_unsafe, loss_history_cbf, loss_history_active, CBF_satisfied_coordinates ):
    grid_points, h_values, grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6, grid_x7, grid_x8, grid_x9 = PlotDataGenerate(h_network)

    plotloss(loss_history, loss_history_safe, loss_history_unsafe, loss_history_cbf, loss_history_active)
    plot_susnums(HARDCOUNT_SAFE, HARDCOUNT_UNSAFE)
    # CBF条件满足情况
    plot_CBF_results(CBF_result, num_Xsafe)
    # 画主图（2D）
    count = 0
    minicount = 0
    for m in range(1, 10):
        for n in range(m+1, 10):
            count += 1
            minicount += 1
            if count%6 == 1:
                minicount = 1
                plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, minicount)
            plot_CISpoints(CBF_satisfied_coordinates, m, n)
            # plot_safe_points(safe_points, m, n)
            # plot2D2(grid_points, h_values, m, n)
            plot2Dnew(h_network, m, n)



def  plotsandaincbf(data):
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1],
                alpha=0.5, s=1)  # alpha透明度，s点大小
    plt.xlabel('X 轴')
    plt.ylabel('Y 轴')
    plt.title('数据点的散点图')
    plt.grid(True, alpha=0.3)
    plt.show()
# 这个函数有问题，实际没用到的线比它的结果会少一些（从图可以看出来）
def useless_lines(h_network):
    parameters = get_parameters(h_network)
    W_all = np.empty((0, 9), float)
    B_all = np.empty((0, 1), float)
    for wb in parameters:
        W_all = np.append(W_all, wb['weight'].detach().numpy(), axis=0)
        B_all = np.append(B_all, [wb['bias'].detach().numpy()], axis=0)
    count = 0
    for i in range(len(B_all)):
        c = W_all[i].T
        A = -np.delete(W_all, i, axis=0)  # 系数矩阵
        b = np.delete(B_all, i, axis=0)  # 右边常数

        x1_bounds = (None, None)
        x2_bounds = (None, None)
        x3_bounds = (None, None)
        x4_bounds = (None, None)
        x5_bounds = (None, None)
        x6_bounds = (None, None)
        x7_bounds = (None, None)
        x8_bounds = (None, None)
        x9_bounds = (None, None)
        # 调用 linprog 求解线性规划问题
        result = linprog(c, A_ub=A, b_ub=b, bounds=[x1_bounds, x2_bounds, x3_bounds, x4_bounds,x5_bounds, x6_bounds, x7_bounds, x8_bounds,x9_bounds], method='highs')
        if result.success:
            wx_plus_b = result.fun + B_all[i]
            if wx_plus_b >= 0:
                count += 1
        else:
            continue

    return count

def save_all_figures(dir_path, fmt='png', dpi=300):
    """
    将当前所有 Matplotlib 图形保存到指定文件夹。

    参数：
      dir_path (str): 要保存到的目录路径，比如 'figures/'
      fmt      (str): 图片格式，默认为 'png'，也可以用 'pdf','jpg' 等
      dpi      (int): 分辨率，默认为 300

    用法示例：
      # ... 你的各种 plt.plot() 操作 ...
      save_all_figures('figures/')
    """
    # 1. 如果文件夹不存在，先创建它
    os.makedirs(dir_path, exist_ok=True)

    # 2. 获取所有图形编号
    fig_nums = plt.get_fignums()
    if not fig_nums:
        print("当前没有可保存的图形。")
        return

    # 3. 遍历保存
    for i, num in enumerate(fig_nums, start=1):
        fig = plt.figure(num)
        filename = os.path.join(dir_path, f'figure_{i}.{fmt}')
        fig.savefig(filename, format=fmt, dpi=dpi, bbox_inches='tight')
        print(f"已保存: {filename}")


# 保存神经网络参数
def export_network_parameters(h_network, filename='cbf_network_params.csv'):
    """将神经网络参数导出为CSV文件，权重和偏置保存在同一文件"""
    parameters = get_parameters(h_network)

    # 导出参数
    params_data = []

    for i, param in enumerate(parameters):
        weight = param['weight'].detach().numpy().flatten()
        bias = param['bias'].detach().numpy().flatten()

        params_data.append({
            'layer': i,
            'w1': weight[0],
            'w2': weight[1],
            'bias': bias[0]
        })

    # 保存为CSV
    pd.DataFrame(params_data).to_csv(filename, index=False)

    print(f"网络参数已保存到 {filename}")


if __name__ == '__main__':

    num_epochs = 1000
    # 系统参数
    T_s = 0.18
    u_min = [-59.3, -59.3, -59.3]  # 控制输入下界
    u_max = [59.3, 59.3, 59.3]  # 控制输入上界
    x_min = np.array([-2, -2, 0, -1, -1, -1, -2.83, -2.83, -2.83])  # 状态x下界
    x_max = np.array([2, 2, 1, 1, 1, 1, 2.83, 2.83, 2.83])  # 状态x上界
    A = np.array([[1, T_s, T_s ** 2 / 2, 0, 0, 0, 0, 0, 0],
                  [0, 1, T_s, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, T_s, T_s ** 2 / 2, 0, 0, 0],
                  [0, 0, 0, 0, 1, T_s, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, T_s, T_s ** 2 / 2],
                  [0, 0, 0, 0, 0, 0, 0, 1, T_s],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]
                  ], dtype=np.float32)
    B = np.array([[T_s ** 3 / 6, 0, 0],
                  [T_s ** 2 / 2, 0, 0],
                  [T_s, 0, 0],
                  [0, T_s ** 3 / 6, 0],
                  [0, T_s ** 2 / 2, 0],
                  [0, T_s, 0],
                  [0, 0, T_s ** 3 / 6],
                  [0, 0, T_s ** 2 / 2],
                  [0, 0, T_s]
                  ], dtype=np.float32)
    # 网络参数
    state_dim = 9
    learning_rate = 9e-3
    # num_epochs = (200+700*4)*10
    num_batches = 1

    # CBF参数
    alpha = 0.1
    # 画图
    # 统计满足真实CBF条件的点数


    plt.close('all')
    CBF_result = []
    # 统计满足必要条件中CBF条件的点数

    # 统计满足CBF条件点的坐标
    CBF_index = []
    CIS_verticles = scipy.io.loadmat('CIS_vertices.mat')['vertices']

    # plot_CIS(CIS_verticles, 1, 2)
    # plt.show()
    # 设置随机种子
    set_seed(57)  # 可以设置任何整数作为种子
    paleto_sus = []
    paleto_cbf = []
    CBF_NUM = []
    SAFE_NUM = []
    UNSAFE_NUM = []
    all_num = []
    safe_name = f'9D-safe-sample-10000.csv'
    unsafe_name = f'9D-unsafe-sample-20000.csv'
    # 数据集生成
    (data_loader_safe, data_loader_unsafe, all_batches_safe, all_batches_unsafe, batch_num_of_safe, batch_num_of_unsafe,
     safe_points, unsafe_points) = dataset_construct(safe_name, unsafe_name)
    num_Xsafe = len(safe_points)
    # 基于数据集构造正多边形

    # w_values, b_values = generate_wb_for_polygon(state_dim = state_dim, data=safe_points, M=num_networks)  # 获取斜率和截距
    # w_values, b_values = generate_polytope_hyperplanes_2d(safe_points)
    # plot2D_cut_scatter(w_values, b_values)
    npoints = 200
    w_values, b_values, gouzaotime = create_9d_simplex_from_data(safe_points, npoints, 0.1)
    # w_values, b_values, npoints = generate_10d_planes_from_9d_convex_hull(safe_points, max_edges=5e5, c=0.1)
    num_networks = npoints

    print(f'n_edges:{npoints}')
    # 初始化网络y

    # 多边形初始化
    h_network = CBFNetwork_Polygon(state_dim=state_dim, w=w_values, b=b_values, num_networks=num_networks)
    # 随机初始化
    # h_network = CBFNetwork_Random(state_dim=state_dim, num_networks=num_networks)
    optimizer = optim.Adam(h_network.parameters(), lr=learning_rate)

    num_paleto = 100

    # 不用权重比
    loss_history, loss_history_safe, loss_history_unsafe, loss_history_cbf, loss_history_active, TOTAL_VALID_ROWS, HARDCOUNT_SAFE, HARDCOUNT_UNSAFE, CBF_satisfied_coordinates, Commom_counts, kept_nall = train_cbf_N(h_network,
                                                                                                                 optimizer,
                                                                                                                 data_loader_safe,
                                                                                                                 data_loader_unsafe,
                                                                                                                 batch_num_of_safe,
                                                                                                                 batch_num_of_unsafe,
                                                                                             )
    num_of_useless = useless_lines(h_network)
    print(f'没用到的线有{num_of_useless}个')

    # plotsandaincbf(CBF_satisfied_coordinates[0])

    plot_result(h_network, safe_points, unsafe_points, HARDCOUNT_SAFE, HARDCOUNT_UNSAFE, num_Xsafe, loss_history_safe, loss_history_unsafe, loss_history_cbf, loss_history_active, CBF_satisfied_coordinates)
    today = date.today()
    save_all_figures(f'C:/Users/XJ_lab/Desktop/实验仿真结果图/{today}-9D-Final/epsilon={epsilon}')
    # save_all_figures(f'C:/Users/XJ_lab/Desktop/实验仿真结果图/CBF_epsilon测试调参/10.2改代码流程/dataset_dim2_2_{iii}/epsilon={epsilon}/随机初始化+熵Loss')
    #  保存神经网络参数
    export_network_parameters(h_network, f'C:/Users/XJ_lab/Desktop/实验仿真结果图/{today}-9D-Final/epsilon={epsilon}'+'csv')
    print(f'9D优化问题求解平均时间：{sum(optimal_times)/len(optimal_times):.4f}')
    print(f'9D训练平均时间：{sum(train_times)/len(train_times):.4f}')
    print(f'9Ddeminate平均时间：{sum(kept_times)/len(kept_times):.4f}')
    print(f'9D构造凸包用时：{gouzaotime:.4f}\n')
    print(f'9D去冗余后平均点数:{kept_nall/len(kept_times):.2f}')
    printfornot = input("要不要画图？Y/N：\n")
    if printfornot == 'Y' or printfornot == 'y':
        plt.show()
