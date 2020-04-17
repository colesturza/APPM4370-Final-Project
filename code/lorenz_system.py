from FORCE import Force
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

linewidth = 3
fontsize = 14
fontweight = 'bold'

# RK4 Integrator
def RK4(dt, t0, tf, V0, dV):

    T = np.arange(t0,tf,dt)

    n = len(T)

    m = V0.shape[0]

    V = np.zeros((n, m))
    V[0] = V0

    for i in range(0, n-1):
        k1 = dt*dV(T[i], V[i])
        k2 = dt*dV(T[i] + dt/2, V[i] + k1/2)
        k3 = dt*dV(T[i] + dt/2, V[i] + k2/2)
        k4 = dt*dV(T[i+1], V[i] + k3)
        V[i+1] = V[i] + (k1 + 2*k2 + 2*k3 + k4)/6

    return T, V

a, r, b = 16, 45, 4

def compute_derivatives(t, state):
    x, y, z = state  # Unpack the state vector
    return np.array([a * (y - x), x * (r - z) - y, x * y - b * z])  # Derivatives

V0 = np.array([0, 1, 2])
T, V = RK4(0.01, 0, 1700, V0, compute_derivatives)

rnn = Force(N=1500,g=1.56,readouts=3)

zt, W_out_mag, x = rnn.fit(V[:140001], 1550, dt=0.01)

fig2, ax2 = plt.subplots()
fig2.set_tight_layout(True)
ax2.plot(T, W_out_mag, lw=linewidth)
ax2.set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
ax2.set_ylabel('|w|', fontsize=fontsize, fontweight=fontweight)
ax2.legend(['|w|'])

simtime, zpt = rnn.predict(x, 1550, 1700, 0.01)
avg_error = rnn.evaluate(x, 1550, 1700, 0.01,  V[140001:])

fig1 = plt.figure()
fig1.set_tight_layout(True)
ax1 = fig1.gca(projection='3d')
ax1.plot(V[140001:], V[140001:], V[140001:], lw=linewidth)
ax1.plot(zpt, zpt, zpt, lw=linewidth)
ax1.set_title('training', fontsize=fontsize, fontweight=fontweight)
ax1.legend(['actual', 'z'])
ax1.set_xlabel('x', fontsize=fontsize, fontweight=fontweight)
ax1.set_ylabel('y', fontsize=fontsize, fontweight=fontweight)
ax1.set_zlabel('z', fontsize=fontsize, fontweight=fontweight)
plt.draw()

plt.show()
