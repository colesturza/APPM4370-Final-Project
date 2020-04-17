from FORCE import Force
import numpy as np
import matplotlib.pyplot as plt

linewidth = 3
fontsize = 14
fontweight = 'bold'

nsecs = 1000
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

a, r, b = 16, 45, 4

def compute_derivatives(t, state):
    x, y, z = state  # Unpack the state vector
    return np.array([a * (y - x), x * (r - z) - y, x * y - b * z])  # Derivatives

V0 = np.array([0, 1, 2])
T, V = RK4(step, 0, nsecs, V0, compute_derivatives)

rnn = Force(N=1500,g=1.56,readouts=1)

lorenz = V[:,0].reshape((V[:,0].shape[0], 1))

zt, W_out_mag, x = rnn.fit(lorenz, nsecs, dt=step)

fig1, axs = plt.subplots(2,1)
fig1.set_tight_layout(True)
line1, = axs[0].plot(T, lorenz, lw=linewidth, c='green')
line2, = axs[0].plot(T, zt, lw=linewidth, c='red')
axs[0].set_title('training', fontsize=fontsize, fontweight=fontweight)
axs[0].legend(['f', 'z'])
axs[0].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[0].set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

line3, = axs[1].plot(T, W_out_mag, lw=linewidth)
axs[1].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[1].set_ylabel('|w|', fontsize=fontsize, fontweight=fontweight)
axs[1].legend(['|w|'])

simtime, zpt = rnn.predict(x, nsecs, nsecs*2, 0.01)
avg_error = rnn.evaluate(x, nsecs, nsecs*2, 0.01, V[:,0])

fig2, ax = plt.subplots()
fig2.set_tight_layout(True)
line1, = ax.plot(simtime, lorenz, lw=linewidth, c='green')
line2, = ax.plot(simtime, zpt, lw=linewidth, c='red')
ax.set_title('testing --- Average Error = {}'.format(avg_error), fontsize=fontsize, fontweight=fontweight)
ax.legend(['f', 'z'])
ax.set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
ax.set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

plt.show()
