from modules.PCA import PCA_Network
from modules.simple_examples import sinwaves
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N = 1000
p = 0.1
g = 1.5
alpha = 1.0
nsecs = 500
dt = 0.1
n_components = 8

def sum_of_four_sinusoids(simtime, *, amp=1.3, freq=1/60):

    ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
         (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + \
         (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + \
         (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
    ft = ft/1.5

    return ft

simtime = np.arange(0, nsecs, dt)
ft = sum_of_four_sinusoids(simtime, amp=1.3, freq=1/60)

rnn = PCA_Network(N=N, p=p, g=g, randomReadout=True)
z_proj, eigvals, proj, proj_w = rnn.fit(simtime, ft, alpha=alpha, n_components=n_components)

############################################################################
# Approximation using leading PCs
lw_f, lw_z = 3.5, 1.5
fig1, ax_fz = plt.subplots()
fig1.set_tight_layout(True)
sns.set_style('white')
sns.despine()

ax_fz.set_xlabel('Time (ms)').set_fontsize('large').set_fontsize('large')
ax_fz.plot(simtime, ft, lw=lw_f, label='f', color='green')
ax_fz.plot(simtime, z_proj, lw=lw_z, label='approximation', color='firebrick')

############################################################################
# Eigenvalues
fig2, ax_eig = plt.subplots()
fig2.set_tight_layout(True)
sns.set_style('white')
sns.despine()

ax_eig.plot(np.arange(1,101), np.log10(eigvals[:100]), color='slateblue')
ax_eig.set_xlabel('eigenvalue').set_fontsize('large')
ax_eig.set_ylabel('log10(eigenvalue)').set_fontsize('large')

# set the x-spine
ax_eig.spines['left'].set_position('zero')

# turn off the right spine/ticks
ax_eig.spines['right'].set_color('none')
ax_eig.yaxis.tick_left()

# set the y-spine
ax_eig.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
ax_eig.spines['top'].set_color('none')
ax_eig.xaxis.tick_bottom()

############################################################################
# Plotting the firing rates of sample neurons
fig3, ax_xs = plt.subplots(n_components, 1, sharex=True)
fig3.set_tight_layout(True)
for i, ax_x in enumerate(ax_xs):
    sns.set_style('white')
    if i < n_components-1:
        sns.despine(ax=ax_x, bottom=True, left=True)
    else:
        sns.despine(ax=ax_x, left=True)
    ax_x.set_yticks([])
    ax_x.plot(simtime, proj[i], color='slateblue')
    ax_x.set_ylabel('PC{}'.format(i+1)).set_fontsize('large')
    ax_x.set_ylim((-25, 25))
ax_xs[-1].set_xlabel('Time (ms)').set_fontsize('large')

############################################################################
plt.show()
