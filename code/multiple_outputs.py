from FORCE import Force
import numpy as np
import matplotlib.pyplot as plt

linewidth = 3
fontsize = 14
fontweight = 'bold'

dt = 0.1
nsecs = 1440
simtime = np.arange(0, nsecs, dt)
simtime2 = np.arange(nsecs, nsecs + nsecs/4, dt)

def func(simtime):
    out = np.zeros((len(simtime), 2))

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

zt, W_out_mag, x = rnn.fit(simtime, func)

fig1, axs = plt.subplots(2,1)
fig1.set_tight_layout(True)
axs[0].plot(simtime, func(simtime)[:,0], lw=linewidth, c='green')
axs[0].plot(simtime, func(simtime)[:,1], lw=linewidth, c='blue')
axs[0].plot(simtime, zt[:,0], lw=linewidth, c='red')
axs[0].plot(simtime, zt[:,1], lw=linewidth, c='orange')
axs[0].set_title('training', fontsize=fontsize, fontweight=fontweight)
axs[0].legend(['$f_1$', '$f_2$', '$z_1$', '$z_2$'])
axs[0].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[0].set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

axs[1].plot(simtime, W_out_mag[:,0], lw=linewidth)
axs[1].plot(simtime, W_out_mag[:,1], lw=linewidth)
axs[1].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[1].set_ylabel('|w|', fontsize=fontsize, fontweight=fontweight)
axs[1].legend(['$|w_1|$', '$|w_2|$'])

zpt = rnn.predict(x, simtime2)
avg_error = rnn.evaluate(x, simtime2, func)

fig2, axs = plt.subplots(2,1)
fig2.set_tight_layout(True)
axs[0].plot(simtime2, func(simtime2)[:,0], lw=linewidth, c='green')
axs[1].plot(simtime2, func(simtime2)[:,1], lw=linewidth, c='blue')
axs[0].plot(simtime2, zpt[:,0], lw=linewidth, c='red')
axs[1].plot(simtime2, zpt[:,1], lw=linewidth, c='orange')
# ax.set_title('training', fontsize=fontsize, fontweight=fontweight)
# ax.legend(['$f_1$', '$f_2$', '$z_1$', '$z_2$'])
# ax.set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
# ax.set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)
# ax.set_title('testing --- Average Error = {}'.format(avg_error), fontsize=fontsize, fontweight=fontweight)

plt.show()
