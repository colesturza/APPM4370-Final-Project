from FORCE import Force
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

linewidth = 1
fontsize = 14
fontweight = 'bold'

nsecs = 100
step = 0.01

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

a, r, b = 10, 28, 8/3

def compute_derivatives(t, state):
    x, y, z = state  # Unpack the state vector
    return np.array([a * (y - x), x * (r - z) - y, x * y - b * z])  # Derivatives

V0 = np.array([0, 1, 2])
T, V = RK4(step, 0, nsecs, V0, compute_derivatives)

simtime_len = len(simtime)

zt, W_out_mag, x = rnn.fit(V, nsecs, dt=step)

rnn = Force(N=1000,g=1.5)

simtime, zpt = rnn.predict(x, nsecs, nsecs*2, step)
avg_error = rnn.evaluate(x, nsecs, nsecs*2, step, V)

fig1, axs = plt.subplots(2,1)
fig1.set_tight_layout(True)
ax1 = fig1.gca(projection='3d')
ax1.plot(V[:,0], V[:,1], V[:,2], lw=linewidth)
ax1.plot(zpt[:,0], zpt[:,1], zpt[:,2], lw=linewidth)
ax1.set_title('training', fontsize=fontsize, fontweight=fontweight)
ax1.legend(['actual', 'z'])
ax1.set_xlabel('x', fontsize=fontsize, fontweight=fontweight)
ax1.set_ylabel('y', fontsize=fontsize, fontweight=fontweight)
ax1.set_zlabel('z', fontsize=fontsize, fontweight=fontweight)
plt.draw()

line3, = axs[1].plot(simtime, W_out_mag, lw=linewidth)
axs[1].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[1].set_ylabel('|w|', fontsize=fontsize, fontweight=fontweight)
axs[1].legend(['|w|'])

simtime, zpt = rnn.predict(x, simtime2[0]+0.01, simtime2[-1]+0.01, 0.01)
avg_error = rnn.evaluate(x, simtime2[0]+0.01, simtime2[-1]+0.01, 0.01, test)

fig2, ax = plt.subplots()
fig2.set_tight_layout(True)
line1, = ax.plot(simtime2, test, lw=linewidth, c='green')
line2, = ax.plot(simtime2, zpt, lw=linewidth, c='red')
ax.set_title('testing --- Average Error = {}'.format(avg_error), fontsize=fontsize, fontweight=fontweight)
ax.legend(['f', 'z'])
ax.set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
ax.set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

# fig, axs = plt.subplots(3,1)
# fig.set_tight_layout(True)
# axs[0].plot(T, V[:,0], lw=linewidth)
# axs[1].plot(T, V[:,1], lw=linewidth)
# axs[2].plot(T, V[:,2], lw=linewidth)

# fig2, ax2 = plt.subplots()
# fig2.set_tight_layout(True)
# ax2.plot(T, W_out_mag, lw=linewidth)
# ax2.set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
# ax2.set_ylabel('|w|', fontsize=fontsize, fontweight=fontweight)
# ax2.legend(['|w|'])
#
# simtime, zpt = rnn.predict(x, 1550, 1700, 0.01)
# avg_error = rnn.evaluate(x, 1550, 1700, 0.01,  V[140001:])
#
# fig1 = plt.figure()
# fig1.set_tight_layout(True)
# ax1 = fig1.gca(projection='3d')
# ax1.plot(V[140001:], V[140001:], V[140001:], lw=linewidth)
# ax1.plot(zpt, zpt, zpt, lw=linewidth)
# ax1.set_title('training', fontsize=fontsize, fontweight=fontweight)
# ax1.legend(['actual', 'z'])
# ax1.set_xlabel('x', fontsize=fontsize, fontweight=fontweight)
# ax1.set_ylabel('y', fontsize=fontsize, fontweight=fontweight)
# ax1.set_zlabel('z', fontsize=fontsize, fontweight=fontweight)
# plt.draw()

plt.show()
