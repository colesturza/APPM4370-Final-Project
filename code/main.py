from FORCE import Force
import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plt

linewidth = 3
fontsize = 14
fontweight = 'bold'

nsecs = 1440
simtime = np.arange(0, nsecs, 0.1) # 0:dt:nsecs-dt;
simtime_len = len(simtime)
#simtime2 = np.arange(1*nsecs, 2*nsecs, dt) # 1*nsecs:dt:2*nsecs-dt;

def func(simtime):
    amp = 1.3
    freq = 1/60
    ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
         (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + \
         (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + \
         (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
    ft = ft/1.5

    return ft

rnn = Force()

zt, W_out_mag = rnn.fit(func, 1440)

fig, axs = plt.subplots(2,1)
fig.set_tight_layout(True)
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

plt.show()
