from modules.PCA import PCA_Network
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import seaborn as sns

N = 1000
p = 0.1
g = 1.5
alpha = 1.0
nsecs = 2500
dt = 0.1
n_components = 8

def sinwaves(simtime, num_waves, amp, freq, noise=False):
    f = np.zeros(len(simtime))

    if noise:
        avgA = sum(amp)/len(amp)
        G = (np.random.randn(len(simtime), num_waves)-0.5)*avgA/4.0
    else:
        G = np.zeros((len(simtime), num_waves))

    for i in range(num_waves):
        f += amp[i]*np.sin(freq[i]*simtime) + G[:,i]

    return f

def triangle(simtime):

    freq = 1/60
    f = sig.sawtooth(simtime*np.pi*freq, width=0.5)

    return f

def periodic(simtime):

    amp = np.array([1, 1/2, 1/3, 1/6])
    freq = np.array([1, 2, 3, 4])*np.pi*(1/60)
    f = sinwaves(simtime, 4, amp, freq)

    return f

def periodic_cmplx(simtime):

    freq = 1/60
    amp = np.array([1, 1/2, 1/6, 1/3,
                    1/8, 1/5, 1/6, 1/10,
                    1/12, 1/4, 1/8, 1/2,
                    1/8, 1/5, 1/6, 1/10])
    freq = np.array([1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 10, 11, 12,
                    13, 14, 15, 16])*np.pi*freq

    f = sinwaves(simtime, 16, amp, freq)

    return f

def discont(simtime):

    freq = 1/60
    f = sig.square(simtime*np.pi*freq)

    return f

def sin(simtime):
    A, F = 1.3, 1/60

    amp = 1*A
    freq = (2/F)*np.pi

    f = sinwaves(simtime, 1, [amp], [freq])

    return f

def multiple(simtime, *argv):

    return np.array( list( zip( *(func(simtime) for func in argv) ) ) )

simtime = np.arange(0, nsecs, dt)
simtime2 = np.arange(nsecs, nsecs+500, dt)

# 2 outputs
ft_2 = multiple(simtime, triangle, periodic_cmplx)
ft_2_2 = multiple(simtime2, triangle, periodic_cmplx)

# 3 outputs
ft_3 = multiple(simtime, triangle, periodic_cmplx, discont)
ft_3_2 = multiple(simtime2, triangle, periodic_cmplx, discont)

# 2 outputs
rnn_2 = PCA_Network(N=N, p=p, g=g, readouts=2, randomReadout=True)

# 3 outputs
rnn_3 = PCA_Network(N=N, p=p, g=g, readouts=3, randomReadout=True)

rnn_2.fit(simtime, ft_2, alpha=alpha)
z_proj_2, eigvals_2, proj_2 = rnn_2.predict(simtime2, n_components=n_components)

rnn_3.fit(simtime, ft_3, alpha=alpha)
z_proj_3, eigvals_3, proj_3 = rnn_3.predict(simtime2, n_components=n_components)

############################################################################
# Approximation using leading PCs 2 outputs
lw_f, lw_z = 3.5, 1.5

fig1, ax_fz_1 = plt.subplots(2, 1, sharex=True)
fig1.subplots_adjust(hspace=0)
sns.set_style('white')

ax_fz_1[0].plot(simtime2, ft_2_2[:,0], lw=lw_f, color='#3cc882')
ax_fz_1[0].plot(simtime2, z_proj_2[:,0], lw=lw_z, color='#ff4f28')
ax_fz_1[0].set_ylabel('Output 1').set_fontsize('large')
ax_fz_1[0].set_yticks(np.arange(-2.5, 2.6, 1))
ax_fz_1[0].set_ylim(-3, 3)
sns.despine(ax=ax_fz_1[0], bottom=True)

ax_fz_1[1].plot(simtime2, ft_2_2[:,1], lw=lw_f, color='#32a064')
ax_fz_1[1].plot(simtime2, z_proj_2[:,1], lw=lw_z, color='#be3b1e')
ax_fz_1[1].set_ylabel('Output 2').set_fontsize('large')
ax_fz_1[1].set_yticks(np.arange(-2.5, 2.6, 1))
ax_fz_1[1].set_ylim(-3, 3)
sns.despine(ax=ax_fz_1[1])

############################################################################
# Approximation using leading PCs 3 outputs
lw_f, lw_z = 3.5, 1.5

fig2, ax_fz_2 = plt.subplots(3, 1, sharex=True)
fig2.subplots_adjust(hspace=0)
sns.set_style('white')

ax_fz_2[0].plot(simtime2, ft_3_2[:,0], lw=lw_f, color='#3cc882')
ax_fz_2[0].plot(simtime2, z_proj_3[:,0], lw=lw_z, color='#ff4f28')
ax_fz_2[0].set_ylabel('Output 1').set_fontsize('large')
ax_fz_2[0].set_yticks(np.arange(-2.5, 2.6, 1))
ax_fz_2[0].set_ylim(-3, 3)
sns.despine(ax=ax_fz_2[0], bottom=True)

ax_fz_2[1].plot(simtime2, ft_3_2[:,1], lw=lw_f, color='#32a064')
ax_fz_2[1].plot(simtime2, z_proj_3[:,1], lw=lw_z, color='#be3b1e')
ax_fz_2[1].set_ylabel('Output 2').set_fontsize('large')
ax_fz_2[1].set_yticks(np.arange(-2.5, 2.6, 1))
ax_fz_2[1].set_ylim(-3, 3)
sns.despine(ax=ax_fz_2[1], bottom=True)

ax_fz_2[2].plot(simtime2, ft_3_2[:,2], lw=lw_f, color='#287346')
ax_fz_2[2].plot(simtime2, z_proj_3[:,2], lw=lw_z, color='#8c2e14')
ax_fz_2[2].set_ylabel('Output 3').set_fontsize('large')
ax_fz_2[2].set_xlabel('Time (ms)').set_fontsize('large')
ax_fz_2[2].set_yticks(np.arange(-2.5, 2.6, 1))
ax_fz_2[2].set_ylim(-3, 3)
sns.despine(ax=ax_fz_2[2])

############################################################################
# Eigenvalues
fig3, ax_eig = plt.subplots()
fig3.set_tight_layout(True)
sns.set_style('white')
sns.despine()

ax_eig.plot(np.arange(1,101), np.log10(eigvals_2[:100]), color='slateblue', label='2 Outputs')
ax_eig.plot(np.arange(1,101), np.log10(eigvals_3[:100]), color='lightskyblue', label='3 Outputs')
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

ax_eig.legend()

############################################################################
# Plotting the firing rates of sample neurons
fig3, ax_xs_1 = plt.subplots(n_components, 1, sharex=True)
fig3.subplots_adjust(hspace=0)
sns.set_style('white')

for i in range(n_components):

    ax_xs_1[i].plot(simtime2, proj_2[i], color='slateblue')
    ax_xs_1[i].set_ylabel('PC{}'.format(i+1)).set_fontsize('large')

    if i < n_components-1:
        sns.despine(ax=ax_xs_1[i], bottom=True, left=True)
    else:
        sns.despine(ax=ax_xs_1[i], left=True)

    ax_xs_1[i].set_ylim((-25, 25))
    ax_xs_1[i].set_yticks([])

ax_xs_1[-1].set_xlabel('Time (ms)').set_fontsize('large')

############################################################################
# Plotting the firing rates of sample neurons
fig4, ax_xs_2 = plt.subplots(n_components, 1, sharex=True)
fig4.subplots_adjust(hspace=0)
sns.set_style('white')

for i in range(n_components):

    ax_xs_2[i].plot(simtime2, proj_3[i], color='slateblue')
    ax_xs_2[i].set_ylabel('PC{}'.format(i+1)).set_fontsize('large')

    if i < n_components-1:
        sns.despine(ax=ax_xs_2[i], bottom=True, left=True)
    else:
        sns.despine(ax=ax_xs_2[i], left=True)

    ax_xs_2[i].set_ylim((-25, 25))
    ax_xs_2[i].set_yticks([])

ax_xs_2[-1].set_xlabel('Time (ms)').set_fontsize('large')

############################################################################
plt.show()
