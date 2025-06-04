"""
学生模板：平方反比引力场中的运动
文件：inverse_square_law_motion_student.py
作者：滕欣玥
日期：2025.6.4

重要：函数名称、参数名称和返回值的结构必须与参考答案保持一致！
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 常量 (如果需要，学生可以自行定义或从参数传入)
# 例如：GM = 1.0 # 引力常数 * 中心天体质量

def derivatives(t, state_vector, gm_val):
    """
    计算状态向量 [x, y, vx, vy] 的导数。

    运动方程（直角坐标系）:
    dx/dt = vx
    dy/dt = vy
    dvx/dt = -GM * x / r^3
    dvy/dt = -GM * y / r^3
    其中 r = sqrt(x^2 + y^2)。

    参数:
        t (float): 当前时间 (solve_ivp 需要，但在此自治系统中不直接使用)。
        state_vector (np.ndarray): 一维数组 [x, y, vx, vy]，表示当前状态。
        gm_val (float): 引力常数 G 与中心天体质量 M 的乘积。

    返回:
        np.ndarray: 一维数组，包含导数 [dx/dt, dy/dt, dvx/dt, dvy/dt]。
    
    实现提示:
    1. 从 state_vector 中解包出 x, y, vx, vy。
    2. 计算 r_cubed = (x**2 + y**2)**1.5。
    3. 注意处理 r_cubed 接近零的特殊情况（例如，如果 r 非常小，可以设定一个阈值避免除以零）。
    4. 计算加速度 ax 和 ay。
    5. 返回 [vx, vy, ax, ay]。
    """
    x, y, vx, vy = state_vector  # 解包当前状态
    r_squared = x**2 + y**2      # 计算 r^2
    r_cubed = r_squared ** 1.5   # 计算 r^3

    # 设置一个最小阈值以避免除以零
    if r_cubed < 1e-12:
        r_cubed = 1e-12

    ax = -gm_val * x / r_cubed  # 牛顿引力加速度 x 分量
    ay = -gm_val * y / r_cubed  # 牛顿引力加速度 y 分量

    return np.array([vx, vy, ax, ay])

def solve_orbit(initial_conditions, t_span, t_eval, gm_val):
    """
    使用 scipy.integrate.solve_ivp 求解轨道运动问题。

    参数:
        initial_conditions (list or np.ndarray): 初始状态 [x0, y0, vx0, vy0]。
        t_span (tuple): 积分时间区间 (t_start, t_end)。
        t_eval (np.ndarray): 需要存储解的时间点数组。
        gm_val (float): GM 值 (引力常数 * 中心天体质量)。

    返回:
        scipy.integrate.OdeSolution: solve_ivp 返回的解对象。
                                     可以通过 sol.y 访问解的数组，sol.t 访问时间点。
    
    实现提示:
    1. 调用 solve_ivp 函数。
    2. `fun` 参数应为你的 `derivatives` 函数。
    3. `args` 参数应为一个元组，包含传递给 `derivatives` 函数的额外参数 (gm_val,)。
    4. 可以选择合适的数值方法 (method)，如 'RK45' (默认) 或 'DOP853'。
    5. 设置合理的相对容差 (rtol) 和绝对容差 (atol) 以保证精度，例如 rtol=1e-7, atol=1e-9。
    """
    sol = solve_ivp(
        fun=derivatives,                 # 微分方程函数
        t_span=t_span,                  # 时间区间
        y0=initial_conditions,          # 初始状态向量
        t_eval=t_eval,                  # 希望返回解的时间点
        args=(gm_val,),                 # 传递额外参数 gm_val
        method='RK45',                  # 使用 Runge-Kutta 方法
        rtol=1e-7, atol=1e-9            # 设置容差以提高精度
    )
    return sol

def calculate_energy(state_vector, gm_val, m=1.0):
    """
    计算质点的（比）机械能。
    （比）能量 E/m = 0.5 * v^2 - GM/r

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        gm_val (float): GM 值。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比能 (E/m)。

    返回:
        np.ndarray or float: （比）机械能。

    实现提示:
    1. 处理 state_vector 可能是一维（单个状态）或二维（多个状态的时间序列）的情况。
    2. 从 state_vector 中提取 x, y, vx, vy。
    3. 计算距离 r = np.sqrt(x**2 + y**2)。注意避免 r=0 导致除以零的错误。
    4. 计算速度的平方 v_squared = vx**2 + vy**2。
    5. 计算比动能 kinetic_energy_per_m = 0.5 * v_squared。
    6. 计算比势能 potential_energy_per_m = -gm_val / r (注意处理 r=0 的情况)。
    7. 比机械能 specific_energy = kinetic_energy_per_m + potential_energy_per_m。
    8. 如果需要总能量，则乘以质量 m。
    """
    state_vector = np.atleast_2d(state_vector)  # 保证是二维数组
    x, y, vx, vy = state_vector[:, 0], state_vector[:, 1], state_vector[:, 2], state_vector[:, 3]

    r = np.sqrt(x**2 + y**2)               # 距离
    r[r < 1e-12] = 1e-12                   # 避免除以零

    v_squared = vx**2 + vy**2              # 速度的平方
    kinetic_energy = 0.5 * v_squared       # 动能/质量
    potential_energy = -gm_val / r         # 势能/质量
    specific_energy = kinetic_energy + potential_energy  # 比机械能

    return specific_energy if len(specific_energy) > 1 else specific_energy[0]


def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算质点的（比）角动量 (z分量)。
    （比）角动量 Lz/m = x*vy - y*vx

    参数:
        state_vector (np.ndarray): 二维数组，每行是 [x, y, vx, vy]，或单个状态的一维数组。
        m (float, optional): 运动质点的质量。默认为 1.0，此时计算的是比角动量 (Lz/m)。

    返回:
        np.ndarray or float: （比）角动量。

    实现提示:
    1. 处理 state_vector 可能是一维或二维的情况。
    2. 从 state_vector 中提取 x, y, vx, vy。
    3. 计算比角动量 specific_Lz = x * vy - y * vx。
    4. 如果需要总角动量，则乘以质量 m。
    """
    state_vector = np.atleast_2d(state_vector)  # 保证二维
    x, y, vx, vy = state_vector[:, 0], state_vector[:, 1], state_vector[:, 2], state_vector[:, 3]

    specific_Lz = x * vy - y * vx  # 比角动量

    return specific_Lz if len(specific_Lz) > 1 else specific_Lz[0]



if __name__ == "__main__":
    # --- 学生可以在此区域编写测试代码或进行实验 ---
    

    # 任务1：实现函数并通过基础测试 (此处不设测试，依赖 tests 文件)

    # 任务2：不同总能量下的轨道绘制
    # 示例：设置椭圆轨道初始条件 (学生需要根据物理意义自行调整或计算得到)

    GM = 1.0#引力常数 * 中心天体质量
    t_start = 0# 初始时间
    t_end = 20# 结束时间
    t_eval = np.linspace(t_start, t_end, 1000)# 时间点数组

    # Task A: Different energy orbits
    initial_conditions = {
        "Elliptical (E<0)": [1.0, 0.0, 0.0, 0.8],# 椭圆轨道
        "Parabolic (E≈0)": [1.0, 0.0, 0.0, np.sqrt(2.0)],# 抛物线轨道
        "Hyperbolic (E>0)": [1.0, 0.0, 0.0, 1.5]# 双曲线轨道
    }

    plt.figure(figsize=(10, 8))
    for label, ic in initial_conditions.items():
        try:
            sol = solve_orbit(ic, (t_start, t_end), t_eval, gm_val=GM)#数值积分求解轨道
            x, y = sol.y[0], sol.y[1]#提取轨道的xy坐标
            energy = calculate_energy(sol.y.T, GM)#计算每个时刻的能量
            angular_momentum = calculate_angular_momentum(sol.y.T)#计算每个时刻的角动量
            print(f"{label}: Initial Energy ≈ {energy[0]:.3f}, Angular Momentum ≈ {angular_momentum[0]:.3f}")#输出初始能量和角动量
            plt.plot(x, y, label=f"{label}")#绘制轨道
        except Exception as e:
            print(f"Failed to simulate {label}: {e}")

    plt.plot(0, 0, 'ko', label='Central Body')#中心天体位置
    plt.xlabel('x Coordinate')
    plt.ylabel('y Coordinate')
    plt.title('Orbits with Different Total Energy')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Task B: Elliptical orbits with different angular momenta
    E_target = -0.5#  # 目标能量 (椭圆轨道的能量)
    rp_values = [0.5, 0.7, 1.0]# 不同的近心点距离 (rp)

    plt.figure(figsize=(10, 8))
    for rp in rp_values:
        vp = np.sqrt(2 * (E_target + GM / rp)) # 计算对应的速度 (vp) 使得能量 E_target 成立
        ic = [rp, 0.0, 0.0, vp]# 初始条件 [x0, y0, vx0, vy0]，其中 x0=rp, y0=0, vx0=0, vy0=vp
        try:
            sol = solve_orbit(ic, (t_start, t_end), t_eval, gm_val=GM)#数值积分求解轨道
            # 提取轨道的 x 和 y 坐标
            x, y = sol.y[0], sol.y[1]
            L = calculate_angular_momentum(sol.y.T)[0]# 计算角动量
            plt.plot(x, y, label=f"rp={rp}, L≈{L:.3f}")
        except Exception as e:
            print(f"Failed to simulate orbit with rp={rp}: {e}")

    plt.plot(0, 0, 'ko', label='Central Body')
    plt.xlabel('x Coordinate')
    plt.ylabel('y Coordinate')
    plt.title('Elliptical Orbits with Different Angular Momenta')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nRefer to 'Project_Instructions.md' for full requirements.")
    print("Use 'tests/test_inverse_square_law_motion.py' to validate your code.")


    # 学生需要根据“项目说明.md”完成以下任务：
    # 1. 实现 `derivatives`, `solve_orbit`, `calculate_energy`, `calculate_angular_momentum` 函数。
    # 2. 针对 E > 0, E = 0, E < 0 三种情况设置初始条件，求解并绘制轨道。
    # 3. 针对 E < 0 且固定时，改变角动量，求解并绘制轨道。
    # 4. (可选) 进行坐标转换和对称性分析。

    
