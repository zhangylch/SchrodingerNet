import jax.numpy as jnp
from jax import lax

def folx_safe_sinc(x):
    """
    终极 Folx 安全版本 Sinc：纯算术掩码 + 泰勒展开
    彻底消灭 jnp.where，解决 select_n 和 __array__() 报错！
    """
    # 1. 构造纯算术的布尔掩码 (转换为 0.0 或 1.0 的浮点数)
    # 只要不用 where，Folx 是完全允许对条件判断做算术追踪的
    is_small = (jnp.abs(x) < 1e-5).astype(x.dtype)
    is_large = 1.0 - is_small
    
    # 2. 核心魔法：绝对安全的分母
    # 当 x 接近 0 时，is_small 是 1.0。
    # 此时 safe_x 会被强行变成 1.0，彻底避开除零导致的 NaN！
    safe_x = x + is_small
    
    # 3. 大值分支：正常的 sinc 计算 (此时 safe_x 绝对不为 0)
    pi_safe_x = jnp.pi * safe_x
    sinc_large = jnp.sin(pi_safe_x) / pi_safe_x
    
    # 4. 小值分支：泰勒展开 (Maclaurin series)
    # 保证在 x=0 处的数值是 1.0，且反向传播梯度绝对正确（为0）
    # sinc(x) ≈ 1 - (pi*x)^2 / 6 + (pi*x)^4 / 120
    pi_x_sq = jnp.square(jnp.pi * x)
    sinc_small = 1.0 - pi_x_sq / 6.0 + jnp.square(pi_x_sq) / 120.0
    
    # 5. 算术拼接！
    # x 很大时：is_large=1, is_small=0，结果等于 sinc_large
    # x 极小时：is_large=0, is_small=1，结果等于 泰勒展开值
    # 全程只有乘法(+)和加法(*)，Folx 追踪起来如丝般顺滑！
    return sinc_large * is_large + sinc_small * is_small
