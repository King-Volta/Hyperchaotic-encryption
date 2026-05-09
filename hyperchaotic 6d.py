import numpy as np
import matplotlib.pyplot as plt

# Parameter (sesuai paper)
a = 10
b = 8/3
c = 28
d = -1
e = 8
r = 3

dt = 0.01
N = 36000   # jumlah iterasi

# Inisialisasi
x1 = np.zeros(N)
x2 = np.zeros(N)
x3 = np.zeros(N)
x4 = np.zeros(N)
x5 = np.zeros(N)
x6 = np.zeros(N)

# initial condition
x1[0], x2[0], x3[0], x4[0], x5[0], x6[0] = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

# Hyperchaotic 6D
def deriv(x):
    x1, x2, x3, x4, x5, x6 = x
    dx1 = a * (x2 - x1) + x4 - x5 - x6
    dx2 = c * x1 - x2 - x1 * x3
    dx3 = -b * x3 + x1 * x2
    dx4 = d * x4 - x2 * x3
    dx5 = e * x6 + x3 * x2
    dx6 = r * x1
    return np.array([dx1, dx2, dx3, dx4, dx5, dx6])

#Runge-kutta 4

def rk4_step(x, dt):
    k1 = deriv(x)
    k2 = deriv(x + 0.5 * dt * k1)
    k3 = deriv(x + 0.5 * dt * k2)
    k4 = deriv(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# =========================
# INTEGRASI RK4
# =========================
for i in range(N-1):
    state = np.array([x1[i], x2[i], x3[i], x4[i], x5[i], x6[i]])
    next_state = rk4_step(state, dt)

    x1[i+1], x2[i+1], x3[i+1], x4[i+1], x5[i+1], x6[i+1] = next_state


# =========================
# BUANG TRANSIENT
# =========================
skip = 1000

x1_p = x1[skip:]
x2_p = x2[skip:]
x3_p = x3[skip:]



# =========================
# VISUALISASI
# =========================

# 1. 3D Attractor (x1, x2, x3)
fig = plt.figure(figsize=(12,5))

ax = fig.add_subplot(121, projection='3d')
ax.plot(x1, x3, x5, linewidth=0.3)
ax.set_title("3D Hyperchaotic Attractor (x1, x3, x5)")
ax.set_xlabel("x1")
ax.set_ylabel("x3")
ax.set_zlabel("x5")


ax = fig.add_subplot(122, projection='3d')
ax.plot(x2, x4, x6, linewidth=0.3)
ax.set_title("3D Hyperchaotic Attractor (x2, x4, x6)")
ax.set_xlabel("x2")
ax.set_ylabel("x4")
ax.set_zlabel("x6")


plt.tight_layout()
plt.show()
