# core/config.py

# --- 核心物理状态基准 ---
ACTIVE_TARGET_V = -69.01
ACTIVE_G_RATIO = 1.77     # 真实测算的电导放大倍数 (R_in 将降至 1/1.77)
REST_TARGET_V = None

# --- In-vivo 实验最优参数 ---
INVIVO_MU_E = 10.5
INVIVO_MU_I = 8.5
INVIVO_SIGMA_RATIO = 0.3
INVIVO_TAU = 50.0