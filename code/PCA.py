from FORCE import Force
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sum_of_four_sinusoids(simtime, *, amp=1.3, freq=1/60):

    ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
         (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + \
         (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + \
         (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
    ft = ft/1.5

    return ft

N = 1000
p = 0.1
g = 1.5
alpha = 1.0
nsecs = 1440
dt = 0.1
learn_every = 2

simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)
simtime2 = np.arange(nsecs, nsecs + nsecs/4, dt)
simtime2_len = len(simtime2)

# simtime = simtime.reshape((simtime_len, 1))
# simtime2 = simtime2.reshape((simtime2_len, 1))

rnn = Force(N=N, p=p, g=g)

_, _, x, corr_mat = rnn.fit(simtime, sum_of_four_sinusoids, alpha=alpha, \
    learn_every=learn_every, return_corr=True)

zpt = rnn.predict(x, simtime2)
avg_error = rnn.evaluate(x, simtime2, sum_of_four_sinusoids)

fig1, axs = plt.subplots(2,1)
fig1.set_tight_layout(True)
fig1.suptitle('Testing -- Average Error = {:.5f}'.format(avg_error))
sns.set_style('white')
sns.despine()

axs[0].plot(simtime2/1000, sum_of_four_sinusoids(simtime2), color='#03adfc')
axs[0].set_xlabel('time')
axs[0].set_ylabel('f')
axs[0].set_title('Actual')

axs[1].plot(simtime2/1000, zpt, color='#fc0345')
axs[1].set_xlabel('time')
axs[1].set_ylabel('z')
axs[1].set_title('Prediction')

# PCA analysis:
eigvals, eigvects = np.linalg.eig(corr_mat)

ind = np.argsort(eigvals)

eigvals, eigvects = eigvals[ind], eigvects[ind]

print(eigvals[-100:])

fig2, ax = plt.subplots()
fig1.set_tight_layout(True)
sns.set_style('white')
sns.despine()
ax.plot(np.arange(1,101), np.log10(eigvals[-100:]), color='#03adfc')
ax.set_xlabel('eigenvalue')
ax.set_ylabel('log10(eigenvalue)')

plt.show()
