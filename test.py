import numpy as np
import matplotlib.pyplot as plt

def approximate_reciprocal(y, k):
    """
    使用混合LUT+泰勒展开方法近似计算 1/y。
    
    Args:
        y (np.ndarray): 输入的整数数组。
        k (int): 用于低位部分 y_L 的比特数。
        
    Returns:
        np.ndarray: 1/y 的近似值数组。
    """
    total_bits = int(np.ceil(np.log2(np.max(y) + 1)))
    num_high_bits = total_bits - k
    
    # 2^k, 用于分解和计算
    two_k = 1 << k

    # 将 y 分解为高位 y_H 和低位 y_L
    # y = y_H * 2^k + y_L
    y_L = y & (two_k - 1)
    y_H = y >> k

    # ---- 核心近似计算 ----
    # 找到 y_H > 0 的索引，只对这些值应用泰勒展开
    # 对于 y_H == 0 的情况（即 y < 2^k），近似误差较大，
    # 在实际应用中通常会用一个小型直接LUT处理。
    # 这里为了验证，我们只分析近似部分。
    valid_indices = np.where(y_H > 0)
    
    y_H_valid = y_H[valid_indices]
    y_L_valid = y_L[valid_indices]

    # 泰勒展开点 a = y_H * 2^k
    a = y_H_valid.astype(np.float64) * two_k

    # 一阶泰勒展开: 1/y ≈ 1/a - y_L / a^2
    approximations_valid = (1.0 / a) - (y_L_valid / (a * a))
    
    # 创建一个与y形状相同的数组来存储最终结果
    approximations = np.zeros_like(y, dtype=np.float64)
    approximations[valid_indices] = approximations_valid
    
    # 对于 y_H == 0 的部分，我们填充真实值，因为这部分不适用该近似
    # 实际MPC中这部分会由一个大小为 2^k 的小LUT覆盖
    invalid_indices = np.where(y_H == 0)
    approximations[invalid_indices] = 1.0 / y[invalid_indices]

    return approximations

# --- 参数设置 ---
TOTAL_BITS = 16
# k: 低位y_L的比特数。这是关键的调节参数。
# 8意味着高8位(y_H)用于LUT索引, 低8位(y_L)用于多项式。
k = 8  

# --- 验证流程 ---
num_high_bits = TOTAL_BITS - k
lut_size = 1 << num_high_bits

print(f"--- 验证参数 ---")
print(f"总比特数: {TOTAL_BITS}")
print(f"低位 k: {k} bits (用于多项式部分)")
print(f"高位: {num_high_bits} bits (用于LUT索引)")
print(f"等效LUT大小: 2^{num_high_bits} = {lut_size}")
print("-" * 20)

# 1. 生成所有可能的16-bit输入值 (y不能为0)
y_values = np.arange(1, 1 << TOTAL_BITS, dtype=np.uint32)

# 2. 计算真实值
ground_truth = 1.0 / y_values

# 3. 计算近似值
approximations = approximate_reciprocal(y_values, k)

# 4. 计算误差 (只在近似有效的范围内)
valid_indices = np.where(y_values >= (1 << k))
absolute_error = np.abs(ground_truth[valid_indices] - approximations[valid_indices])
relative_error = absolute_error / ground_truth[valid_indices]

# 5. 分析并打印结果
max_abs_error = np.max(absolute_error)
max_rel_error = np.max(relative_error)

# 找到误差最大的点
idx_max_abs_error = np.argmax(absolute_error)
y_at_max_abs_error = y_values[valid_indices][idx_max_abs_error]

idx_max_rel_error = np.argmax(relative_error)
y_at_max_rel_error = y_values[valid_indices][idx_max_rel_error]


print("\n--- 误差分析结果 ---")
print(f"最大绝对误差: {max_abs_error:.10f}")
print(f"  (发生在 y = {y_at_max_abs_error})")
print(f"最大相对误差: {max_rel_error:.4%} ({max_rel_error:.10f})")
print(f"  (发生在 y = {y_at_max_rel_error})")

# 理论误差上界 (发生在 y_H=1 时, 即 y_values 在 2^k 附近)
# |E| < 1 / (y_H^3 * 2^k), y_H=1
theoretical_bound = 1.0 / (1**3 * (1 << k))
y_approx_point = 1 << k
theoretical_error_approx = (( (1<<k) -1)**2) / ((1 * (1<<k))**3)

print(f"\n理论误差上界 (在 y ≈ {y_approx_point} 附近): < {theoretical_bound:.10f}")
print("注：实际最大误差应略小于此理论值。")


# --- 6. 绘图 ---
plt.figure(figsize=(12, 6))

# 使用对数坐标可以更好地观察小y值处的误差
plt.plot(y_values[valid_indices], absolute_error, label='Absolute Error', alpha=0.8)
plt.xscale('log')
plt.yscale('log')
plt.title(f'Absolute Error of 1/y Approximation (k={k})')
plt.xlabel('Input y (log scale)')
plt.ylabel('Absolute Error (log scale)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.axvline(x=y_at_max_abs_error, color='r', linestyle='--', label=f'Max Error at y={y_at_max_abs_error}')
plt.legend()

plt.tight_layout()
plt.show()