#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：二阶常微分方程边值问题数值解法 - 学生代码模板

本项目要求实现两种数值方法求解边值问题：
1. 有限差分法 (Finite Difference Method)
2. scipy.integrate.solve_bvp 方法

问题设定：
y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
边界条件：y(0) = 0, y(5) = 3

学生需要完成所有标记为 TODO 的函数实现。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve

# ============================================================================ 
# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================

def solve_bvp_finite_difference(n):
    """
    使用有限差分法求解二阶常微分方程边值问题。
    方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
    边界条件：y(0) = 0, y(5) = 3
    """
    # 1. 网格
    a, b = 0, 5
    h = (b - a) / (n + 1)
    x = np.linspace(a, b, n + 2)  # 包含边界点
    # 2. 系数矩阵和右端项
    A = np.zeros((n, n))
    b_vec = np.zeros(n)
    for i in range(n):
        xi = x[i + 1]
        sinx = np.sin(xi)
        expx = np.exp(xi)
        # 三对角系数
        if i > 0:
            A[i, i - 1] = (1 / h**2) - (sinx / (2 * h))
        A[i, i] = (-2 / h**2) + expx
        if i < n - 1:
            A[i, i + 1] = (1 / h**2) + (sinx / (2 * h))
        # 右端项
        b_vec[i] = xi**2
    # 3. 边界条件
    # y_0 = 0, y_{n+1} = 3
    b_vec[0] -= ((1 / h**2) - (np.sin(x[1]) / (2 * h))) * 0  # y_0 = 0
    b_vec[-1] -= ((1 / h**2) + (np.sin(x[-2]) / (2 * h))) * 3  # y_{n+1} = 3
    # 4. 求解线性方程组
    y_inner = solve(A, b_vec)
    # 5. 拼接边界
    y_full = np.zeros(n + 2)
    y_full[0] = 0
    y_full[1:-1] = y_inner
    y_full[-1] = 3
    return x, y_full

# ============================================================================ 
# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    为 scipy.integrate.solve_bvp 定义ODE系统。
    """
    dy0 = y[1]
    dy1 = -np.sin(x) * y[1] - np.exp(x) * y[0] + x**2
    return np.vstack((dy0, dy1))

def boundary_conditions_for_solve_bvp(ya, yb):
    """
    为 scipy.integrate.solve_bvp 定义边界条件。
    """
    return np.array([ya[0] - 0, yb[0] - 3])

def solve_bvp_scipy(n_initial_points=11):
    """
    使用 scipy.integrate.solve_bvp 求解BVP。
    """
    x_init = np.linspace(0, 5, n_initial_points)
    # 初始猜测：线性插值
    y_init = np.zeros((2, n_initial_points))
    y_init[0] = 3 * x_init / 5  # y(x) 线性猜测
    y_init[1] = 3 / 5  # y'(x) 线性猜测
    sol = solve_bvp(ode_system_for_solve_bvp, boundary_conditions_for_solve_bvp, x_init, y_init)
    if not sol.success:
        raise RuntimeError("solve_bvp failed: " + sol.message)
    return sol.x, sol.y[0]

# ============================================================================ 
# 主程序：测试和比较两种方法
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("二阶常微分方程边值问题数值解法比较")
    print("方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2")
    print("边界条件：y(0) = 0, y(5) = 3")
    print("=" * 60)
    
    # 设置参数
    n_points = 50  # 有限差分法的内部网格点数
    
    try:
        # 方法1：有限差分法
        print("\n1. 有限差分法求解...")
        x_fd, y_fd = solve_bvp_finite_difference(n_points)
        print(f"   网格点数：{len(x_fd)}")
        print(f"   y(0) = {y_fd[0]:.6f}, y(5) = {y_fd[-1]:.6f}")
        
    except NotImplementedError:
        print("   有限差分法尚未实现")
        x_fd, y_fd = None, None
    
    try:
        # 方法2：scipy.integrate.solve_bvp
        print("\n2. scipy.integrate.solve_bvp 求解...")
        x_scipy, y_scipy = solve_bvp_scipy()
        print(f"   网格点数：{len(x_scipy)}")
        print(f"   y(0) = {y_scipy[0]:.6f}, y(5) = {y_scipy[-1]:.6f}")
        
    except NotImplementedError:
        print("   solve_bvp 方法尚未实现")
        x_scipy, y_scipy = None, None
    
    # 绘图比较
    plt.figure(figsize=(12, 8))
    
    # 子图1：解的比较
    plt.subplot(2, 1, 1)
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-o', markersize=3, label='Finite Difference Method', linewidth=2)
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', label='scipy.integrate.solve_bvp', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Comparison of Numerical Solutions for BVP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：解的差异（如果两种方法都实现了）
    plt.subplot(2, 1, 2)
    if (x_fd is not None and y_fd is not None and 
        x_scipy is not None and y_scipy is not None):
        
        # 将 scipy 解插值到有限差分网格上进行比较
        y_scipy_interp = np.interp(x_fd, x_scipy, y_scipy)
        difference = np.abs(y_fd - y_scipy_interp)
        
        plt.semilogy(x_fd, difference, 'g-', linewidth=2, label='|Finite Diff - solve_bvp|')
        plt.xlabel('x')
        plt.ylabel('Absolute Difference')
        plt.title('Absolute Difference Between Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 数值比较
        max_diff = np.max(difference)
        mean_diff = np.mean(difference)
        print(f"\n数值比较：")
        print(f"   最大绝对误差：{max_diff:.2e}")
        print(f"   平均绝对误差：{mean_diff:.2e}")
    else:
        plt.text(0.5, 0.5, 'Need both methods implemented\nfor comparison', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Difference Plot (Not Available)')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=" * 60)
    print("实验完成！")
    print("请在实验报告中分析两种方法的精度、效率和适用性。")
    print("=" * 60)
