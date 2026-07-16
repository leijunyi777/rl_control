import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# 1. 定义时变环境参数 (正弦波)
# ============================================================
A = 1.5                 # Amplitude of environmental fluctuation
freq = 0.15             # Frequency

def mu_func(t):
    """Environmental input (e.g., gap size in traffic)"""
    return A * np.sin(2 * np.pi * freq * t)

# ============================================================
# 2. 定义两个微分方程
# ============================================================
eps = 0.3               # Time constant (response speed)

# --- Model 1: Original simple quadratic (Paper 2) ---
def ode_original(t, z):
    mu = mu_func(t)
    dzdt = (1.0 / eps) * z * (mu - z)
    return dzdt

# --- Model 2: New tanh-saturated design (Inspired by Paper 1) ---
def ode_saturated(t, z):
    mu = mu_func(t)
    
    # Parameters from Paper 1 philosophy
    d = 0.15            # Inertia / resistance (keeps memory)
    u_att = 1.2         # Attention knob (social influence strength)
    k = 2.5             # Steepness of tanh (tunable sensitivity)
    b = 0.0             # External bias (e.g., navigation command)
    
    # Saturated environmental opportunity
    env_sat = np.tanh(k * (mu + b))
    # Saturated self-inhibition
    self_sat = np.tanh(k * z)
    
    # Core drive: z * (saturated input - saturated self)
    social_drive = z * (env_sat - self_sat)
    
    # Total change: inertia + attention-scaled social drive
    dzdt = (1.0 / eps) * (-d * z + u_att * social_drive)
    return dzdt

# ============================================================
# 3. 数值积分
# ============================================================
t_span = (0, 30)
t_eval = np.linspace(0, 30, 3000)

# Initial condition (slightly perturbed to break the deadlock)
z0 = [0.1]

sol_orig = solve_ivp(ode_original, t_span, z0, t_eval=t_eval, method='RK45')
sol_sat = solve_ivp(ode_saturated, t_span, z0, t_eval=t_eval, method='RK45')

# ============================================================
# 4. 绘图 (All labels in English)
# ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# ---- Subplot 1: Environmental input ----
ax1.plot(t_eval, mu_func(t_eval), color='blue', linewidth=2, label=r'$\mu(t)$ = Environmental input')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax1.set_ylabel(r'Input $\mu(t)$', fontsize=12)
ax1.set_title('Fig 1: Time-varying environmental trigger (sinusoidal)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# ---- Subplot 2: Original simple model ----
ax2.plot(t_eval, sol_orig.y[0], color='red', linewidth=2, 
         label=r'Original: $\dot{z} = z(\mu - z)$')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_ylabel(r'Opinion $z$', fontsize=12)
ax2.set_title('Fig 2: Original model (No saturation, No inertia)', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
# Annotate the sharp explosion
ax2.text(5, 1.3, 'Unbounded growth', color='darkred', fontsize=9, ha='center')
ax2.text(22, -0.8, 'Instant drop to 0', color='darkred', fontsize=9, ha='center')

# ---- Subplot 3: New Saturated model ----
ax3.plot(t_eval, sol_sat.y[0], color='darkorange', linewidth=2.5,
         label=r'Saturated: $\dot{z} = -d z + u \cdot z \cdot [\tanh(k\mu) - \tanh(kz)]$')
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_xlabel('Time (seconds)', fontsize=12)
ax3.set_ylabel(r'Opinion $z$', fontsize=12)
ax3.set_title('Fig 3: New tanh-saturated model (With inertia & tunable sensitivity)', fontsize=13)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')
# Annotate the smooth saturation
ax3.text(5, 0.9, 'Saturated peak\n(no explosion)', color='darkgreen', fontsize=9, ha='center')
ax3.text(22, -0.5, 'Slow decay due\nto inertia (-d·z)', color='darkgreen', fontsize=9, ha='center')

plt.tight_layout()
plt.show()