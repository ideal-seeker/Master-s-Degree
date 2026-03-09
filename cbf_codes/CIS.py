import numpy as np


# 示例的 Pre 函数（定义如何计算 Pre(Ω)）
def Pre(omega, A, B, U):
    """预像（前驱集）计算"""
    pre_omega = []
    for x in omega:
        for u in U:
            # 这里假设系统动态为 x' = A * x + B * u
            # 确保 u 为列向量
            u = np.array(u).reshape(-1, 1)  # 转换为列向量
            x_next = A @ x + B @ u  # 确保 x 和 u 的维度匹配
            pre_omega.append(x_next.flatten())  # 展平以保持一致的输出格式
    return np.array(pre_omega)


def compute_C_infinity(g, X, U, A, B, max_iter=30):
    """
    计算C∞（最大控制不变集）

    :param g: 定义系统动力学的函数
    :param X: 状态空间集合
    :param U: 控制输入空间
    :param A: 系统矩阵A
    :param B: 系统矩阵B
    :param max_iter: 最大迭代次数
    :return: 控制不变集 C∞
    """
    # 初始化Ω0
    Omega_k = X
    k = -1

    while k < max_iter:
        k += 1
        # 计算Ωk+1 = Pre(Ωk) ∩ Ωk
        Omega_k_next = Pre(Omega_k, A, B, U)

        # 计算Ωk+1和Ωk的交集
        Omega_k_next = np.intersect1d(Omega_k_next, Omega_k, axis=0)

        # 如果Ωk+1 == Ωk，停止迭代
        if np.array_equal(Omega_k_next, Omega_k):
            break

        # 更新Ωk
        Omega_k = Omega_k_next

    C_infinity = Omega_k_next  # 输出最大控制不变集
    return C_infinity


# 示例输入
A = np.array([[2, 1], [-1, 2]])  # 系统矩阵 A
B = np.array([[1, 0], [0, 1]])  # 系统矩阵 B
X = np.array([[-1, -1], [1, 1]])  # 状态空间集合 X
U = np.array([[-1], [1]])  # 控制输入空间集合 U (确保是列向量)

# 计算C∞
C_infinity = compute_C_infinity(None, X, U, A, B, max_iter=30)
print("C∞: ", C_infinity)