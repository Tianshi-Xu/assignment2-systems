import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions ---

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def silu_derivative(x):
    sig_x = sigmoid(x)
    return sig_x * (1.0 + x * (1.0 - sig_x))

# --- Approximation Implementations ---

def approx_reciprocal(y, k):
    """为 f(y) = 1/y 应用混合近似法"""
    total_bits = int(np.ceil(np.log2(np.max(y) + 1)))
    two_k = 1 << k
    y_L = y & (two_k - 1)
    y_H = y >> k
    
    approximations = np.zeros_like(y, dtype=np.float64)
    valid_indices = np.where(y_H > 0)
    
    y_H_valid = y_H[valid_indices]
    y_L_valid = y_L[valid_indices]
    
    a = y_H_valid.astype(np.float64) * two_k
    # 泰勒展开: f(a) + f'(a)(y-a) = 1/a - 1/a^2 * y_L
    approximations[valid_indices] = (1.0 / a) - (y_L_valid / (a * a))
    
    # 小于2^k的值在实际中由小LUT覆盖，此处填充真实值以作区分
    invalid_indices = np.where(y_H == 0)
    approximations[invalid_indices] = 1.0 / y[invalid_indices]
    
    return approximations

def approx_exp(x_int, k, f_bits):
    """为 f(x) = e^x 应用混合近似法 (输入为定点数)"""
    scale = 2.0**(-f_bits)
    two_k = 1 << k
    
    # 需要处理有符号整数的位操作
    x_L = x_int & (two_k - 1)
    x_H = x_int >> k

    approximations = np.zeros_like(x_int, dtype=np.float64)
    
    # 展开点a的值
    a_val = (x_H.astype(np.float64) * two_k) * scale
    # 低位x_L的值
    x_L_val = x_L.astype(np.float64) * scale

    # LUT查找的值: C = f(a) = f'(a) = e^a
    C = np.exp(a_val)
    
    # 泰勒展开: f(a) + f'(a)(x-a) = C + C * x_L_val = C * (1 + x_L_val)
    approximations = C * (1 + x_L_val)
    
    return approximations
    
def approx_silu(x_int, k, f_bits):
    """为 f(x) = SiLU(x) 应用混合近似法 (输入为定点数)"""
    scale = 2.0**(-f_bits)
    two_k = 1 << k
    
    x_L = x_int & (two_k - 1)
    x_H = x_int >> k

    approximations = np.zeros_like(x_int, dtype=np.float64)
    
    a_val = (x_H.astype(np.float64) * two_k) * scale
    x_L_val = x_L.astype(np.float64) * scale
    
    # LUT查找的值
    lut_A = silu(a_val)
    lut_B = silu_derivative(a_val)
    
    # 泰勒展开: f(a) + f'(a)(x-a) = lut_A + lut_B * x_L_val
    approximations = lut_A + lut_B * x_L_val
    
    return approximations

# --- Main Test Runner ---

def run_test(func_name, total_bits, k, f_bits=0):
    """
    执行一次完整的测试并打印报告。
    
    Args:
        func_name (str): 'reciprocal', 'exp', or 'silu'
        total_bits (int): 8 or 16
        k (int): 低位的比特数
        f_bits (int): 定点数的小数位数 (仅用于exp和silu)
    """
    print(f"\n{'='*50}")
    print(f"Running test for: {func_name.upper()}")
    print(f"Input Bit-width: {total_bits}, Low-bit k: {k}")
    if f_bits > 0:
        print(f"Fixed-point format: Q{total_bits - f_bits}.{f_bits}")
    print(f"{'='*50}")
    
    # 1. 生成输入数据
    if func_name == 'reciprocal':
        # 无符号正整数
        x_int = np.arange(1, 1 << total_bits, dtype=np.uint32)
        ground_truth = 1.0 / x_int
        approximations = approx_reciprocal(x_int, k)
        valid_indices = np.where(x_int >= (1 << k)) # 近似有效的范围
    
    elif func_name == 'exp':
        # Softmax场景下的有符号非正整数
        min_val = -(1 << (total_bits - 1))
        max_val = 0
        x_int = np.arange(min_val, max_val + 1, dtype=np.int32)
        x_float = x_int * (2.0**(-f_bits))
        ground_truth = np.exp(x_float)
        approximations = approx_exp(x_int, k, f_bits)
        valid_indices = slice(None) # 近似对所有范围都应用

    elif func_name == 'silu':
        # 通用场景下的有符号整数
        min_val = -(1 << (total_bits - 1))
        max_val = (1 << (total_bits - 1)) -1
        x_int = np.arange(min_val, max_val + 1, dtype=np.int32)
        x_float = x_int * (2.0**(-f_bits))
        ground_truth = silu(x_float)
        approximations = approx_silu(x_int, k, f_bits)
        valid_indices = slice(None)

    else:
        raise ValueError("Unknown function name")

    # 2. 计算误差
    abs_error = np.abs(ground_truth[valid_indices] - approximations[valid_indices])
    # 防止除以0
    rel_error = np.divide(abs_error, np.abs(ground_truth[valid_indices]), 
                          out=np.zeros_like(abs_error), 
                          where=np.abs(ground_truth[valid_indices]) > 1e-9)

    # 3. 打印报告
    max_abs_error = np.max(abs_error)
    idx_max_abs = np.argmax(abs_error)
    x_at_max_abs = x_int[valid_indices][idx_max_abs]
    
    max_rel_error = np.max(rel_error)
    idx_max_rel = np.argmax(rel_error)
    x_at_max_rel = x_int[valid_indices][idx_max_rel]
    
    ### NEW ###
    # 获取最大相对误差点的真实值和近似值
    truth_at_max_rel = ground_truth[valid_indices][idx_max_rel]
    approx_at_max_rel = approximations[valid_indices][idx_max_rel]
    ### END NEW ###

    print("--- Error Analysis Report ---")
    print(f"Max Absolute Error: {max_abs_error:.8f} (at int value: {x_at_max_abs})")
    print(f"Max Relative Error: {max_rel_error:.4%} (at int value: {x_at_max_rel})")
    ### NEW ###
    print(f"  - Ground Truth at this point : {truth_at_max_rel:.8f}")
    print(f"  - Approximation at this point: {approx_at_max_rel:.8f}")
    ### END NEW ###
    
    # 4. 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(x_int[valid_indices], abs_error, label='Absolute Error', alpha=0.8)
    if func_name == 'reciprocal':
        plt.xscale('log')
        plt.yscale('log')
    plt.title(f'Absolute Error for {func_name.upper()} (bits={total_bits}, k={k})')
    plt.xlabel('Input Integer Value')
    plt.ylabel('Absolute Error')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

# --- 配置要运行的测试 ---
if __name__ == '__main__':
    # (函数名, 总比特数, 低位k, [小数位数f_bits])
    TESTS_TO_RUN = [
        ("reciprocal", 16, 4),
        ("exp", 16, 8, 8),        # 16-bit, Q8.8 format
        ("silu", 16, 8, 8),       # 16-bit, Q8.8 format
        ("silu", 8, 4, 4)         # 8-bit, Q4.4 format
    ]
    
    for test_config in TESTS_TO_RUN:
        run_test(*test_config)