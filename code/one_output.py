from modules.FORCE import Force
import numpy as np
import matplotlib.pyplot as plt

linewidth = 3
fontsize = 14
fontweight = 'bold'

dt = 0.1
nsecs = 1000
simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)
simtime2 = np.arange(1*nsecs, 2*nsecs, dt)

def func(simtime):
    amp = 1.3
    freq = 1/60
    ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime)
    ft = ft/1.5
    return ft.reshape((len(simtime),1))

rnn = Force(g=1.25)

zt, W_out_mag, x = rnn.fit(simtime, func)

fig1, axs = plt.subplots(2,1)
fig1.set_tight_layout(True)
line1, = axs[0].plot(simtime, func(simtime), lw=linewidth, c='green')
line2, = axs[0].plot(simtime, zt, lw=linewidth, c='red')
axs[0].set_title('training', fontsize=fontsize, fontweight=fontweight)
axs[0].legend(['f', 'z'])
axs[0].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[0].set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

line3, = axs[1].plot(simtime, W_out_mag, lw=linewidth)
axs[1].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[1].set_ylabel('|w|', fontsize=fontsize, fontweight=fontweight)
axs[1].legend(['|w|'])

zpt = rnn.predict(x, simtime2)
avg_error = rnn.evaluate(x, simtime2, func)

fig2, ax = plt.subplots()
fig2.set_tight_layout(True)
line1, = ax.plot(simtime2, func(simtime2), lw=linewidth, c='green')
line2, = ax.plot(simtime2, zpt, lw=linewidth, c='red')
ax.set_title('testing -- Average Error = {:.5f}'.format(avg_error), fontsize=fontsize, fontweight=fontweight)
ax.legend(['f', 'z'])
ax.set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
ax.set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

plt.show()
