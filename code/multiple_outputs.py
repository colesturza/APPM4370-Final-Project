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

    simtime_len = len(simtime)
    out = np.zeros((simtime_len,2))

    amp = 1
    freq = 1/60
    out[:,0] = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime)
    out[:,0] = out[:,0]/1.5

    amp = 1
    freq = 1/60
    phase = np.pi/2
    out[:,1] = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime + phase)
    out[:,1] = out[:,1]/1.5

    return out

rnn = Force(g=1.25,readouts=2)

zt, W_out_mag, x = rnn.fit(func, 1440)

fig1, axs = plt.subplots(2,1)
fig1.set_tight_layout(True)
line1, = axs[0].plot(simtime, func(simtime)[:,0], lw=linewidth, c='green')
line2, = axs[0].plot(simtime, func(simtime)[:,1], lw=linewidth, c='blue')
line3, = axs[0].plot(simtime, zt[:,0], lw=linewidth, c='red')
line4, = axs[0].plot(simtime, zt[:,1], lw=linewidth, c='orange')
axs[0].set_title('training', fontsize=fontsize, fontweight=fontweight)
axs[0].legend(['$f_1$', '$f_2$', '$z_1$', '$z_2$'])
axs[0].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[0].set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

line5, = axs[1].plot(simtime, W_out_mag[:,0], lw=linewidth)
line6, = axs[1].plot(simtime, W_out_mag[:,1], lw=linewidth)
axs[1].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[1].set_ylabel('|w|', fontsize=fontsize, fontweight=fontweight)
axs[1].legend(['$|w_1|$', '$|w_2|$'])

simtime, zpt = rnn.predict(x, 1440, 2880, 0.1)
avg_error = rnn.evaluate(x, 1440, 2880, 0.1, func)

fig2, ax = plt.subplots()
fig2.set_tight_layout(True)
line1, = ax.plot(simtime, func(simtime)[:,0], lw=linewidth, c='green')
line2, = ax.plot(simtime, func(simtime)[:,1], lw=linewidth, c='blue')
line3, = ax.plot(simtime, zt[:,0], lw=linewidth, c='red')
line4, = ax.plot(simtime, zt[:,1], lw=linewidth, c='orange')
ax.set_title('training', fontsize=fontsize, fontweight=fontweight)
ax.legend(['$f_1$', '$f_2$', '$z_1$', '$z_2$'])
ax.set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
ax.set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)
ax.set_title('testing --- Average Error = {}'.format(avg_error), fontsize=fontsize, fontweight=fontweight)

plt.show()
