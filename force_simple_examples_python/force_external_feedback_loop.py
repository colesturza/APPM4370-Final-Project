# FORCE.m
#
# This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1A with the RLS
# learning rule.
#
# written by David Sussillo originally in matlab

import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plt

linewidth = 3
fontsize = 14
fontweight = 'bold'

N = 1000
p = 0.1
g = 1.5           # g greater than 1 leads to chaotic networks.
alpha = 1.0
nsecs = 1440
dt = 0.1
learn_every = 2

scale = 1.0/np.sqrt(p*N)
M = random(N, N, density=p, data_rvs=np.random.randn) * g * scale
M = M.toarray()

nRec2Out = N
wo = np.zeros(nRec2Out)
dw = np.zeros(nRec2Out)
wf = 2.0*(np.random.rand(N)-0.5)

print('   N:           {}'.format(N))
print('   g:           {}'.format(g))
print('   p:           {}'.format(p))
print('   nRec2Out:    {}'.format(nRec2Out))
print('   alpha:       {:.3f}'.format(alpha))
print('   nsecs:       {}'.format(nsecs))
print('   learn_every: {}'.format(learn_every))

simtime = np.arange(0, nsecs, dt) # 0:dt:nsecs-dt;
simtime_len = len(simtime)
simtime2 = np.arange(1*nsecs, 2*nsecs, dt) # 1*nsecs:dt:2*nsecs-dt;

amp = 1.3
freq = 1/60
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
     (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + \
     (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + \
     (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
ft = ft/1.5

ft2 = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime2) + \
      (amp/2.0)*np.sin(2.0*np.pi*freq*simtime2) + \
      (amp/6.0)*np.sin(3.0*np.pi*freq*simtime2) + \
      (amp/3.0)*np.sin(4.0*np.pi*freq*simtime2)
ft2 = ft2/1.5

wo_len = np.zeros(simtime_len)
zt = np.zeros(simtime_len)
zpt = np.zeros(simtime_len)
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn()

x = x0
r = np.tanh(x)
z = z0

fig, axs = plt.subplots(2,1, figsize=(20,20))
P = (1.0/alpha)*np.eye(nRec2Out)
for ti, t in enumerate(simtime):

    if (ti+1) % (nsecs/2) == 0:
        print('time: {:.3f}.'.format(t))
        axs[0].plot(simtime, ft, lw=linewidth, c='green')
        axs[0].plot(simtime, zt, lw=linewidth, c='red')
        axs[0].set_title('training', fontsize=fontsize, fontweight=fontweight)
        axs[0].legend(['f', 'z'])
        axs[0].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
        axs[0].set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

        axs[1].plot(simtime, wo_len, lw=linewidth)
        axs[1].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
        axs[1].set_ylabel('|w|', fontsize=fontsize, fontweight=fontweight)
        axs[1].legend(['|w|'])

    # sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M.dot(r*dt) + wf*(z*dt)
    r = np.tanh(x)
    z = wo.dot(r)

    if (ti+1) % learn_every == 0:
        # update inverse correlation matrix
        k = P.dot(r)
        rPr = r.dot(k)
        c = 1.0/(1.0 + rPr)
        P = P - np.outer(k, k * c)

        # update the error for the linear readout
        e = z - ft[ti]

        # update the output weights
        dw = -e * k * c

        wo = wo + dw

    # Store the output of the system.
    zt[ti] = z
    wo_len[ti] = np.sqrt(wo.dot(wo))

plt.show()

error_avg = np.sum(np.abs(zt-ft))/simtime_len
print('Training MAE: {:.3f}'.format(error_avg))
print('Now testing... please wait.')

# Now test.
for ti, t in enumerate(simtime):				# don't want to subtract time in indices

    # sim, so x(t) and r(t) are created.
    x = (1.0-dt)*x + M.dot(r*dt) + wf*(z*dt)
    r = np.tanh(x)
    z = wo.dot(r)

    zpt[ti] = z

error_avg = np.sum(np.abs(zpt-ft2))/simtime_len
print('Testing MAE: {:.3f}'.format(error_avg))

fig, axs = plt.subplots(2,1, figsize=(20,20))
axs[0].plot(simtime, ft, lw=linewidth, c='green')
axs[0].plot(simtime, zt, lw=linewidth, c='red')
axs[0].set_title('training', fontsize=fontsize, fontweight=fontweight)
axs[0].legend(['f', 'z'])
axs[0].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[0].set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

axs[1].plot(simtime, ft2, lw=linewidth, c='green')
axs[1].plot(simtime, zpt, lw=linewidth, c='red')
axs[1].set_title('testing', fontsize=fontsize, fontweight=fontweight)
axs[1].legend(['f', 'z'])
axs[1].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
axs[1].set_ylabel('f and z', fontsize=fontsize, fontweight=fontweight)

plt.show()
