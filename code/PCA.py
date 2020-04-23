from modules.PCA import PCA_NN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
# simtime2 = np.arange(nsecs, nsecs + nsecs/4, dt)
# simtime2_len = len(simtime2)

rnn = PCA_NN(N=N, p=p, g=g)

# zt, x, eigvals, projections = rnn.fit(simtime, sum_of_four_sinusoids, alpha=alpha, learn_every=learn_every)

# zpt = rnn.predict(x, simtime2)
# avg_error = rnn.evaluate(x, simtime2, sum_of_four_sinusoids)

# fig1, axs = plt.subplots(2,1)
# fig1.set_tight_layout(True)
# fig1.suptitle('Testing -- Average Error = {:.5f}'.format(avg_error))
# sns.set_style('white')
# sns.despine()
#
# axs[0].plot(simtime2, sum_of_four_sinusoids(simtime2), color='#233D4D')
# axs[0].set_xlabel('time')
# axs[0].set_ylabel('f')
# axs[0].set_title('Actual')
#
# axs[1].plot(simtime2, zpt, color='#FE7F2D')
# axs[1].set_xlabel('time')
# axs[1].set_ylabel('z')
# axs[1].set_title('Prediction')

eigvals = pd.read_csv('eigvals.txt', header=None).to_numpy()
projections = pd.read_csv('projections.txt', sep=' ', header=None).to_numpy()

fig2, ax2 = plt.subplots()
fig2.set_tight_layout(True)
sns.set_style('white')
sns.despine()
ax2.plot(np.arange(1,101), np.log10(eigvals), color='#443742')
ax2.plot([0, 100], [0, 0], color='k')
ax2.set_xlabel('eigenvalue')
ax2.set_ylabel('log10(eigenvalue)')

fig3, ax3 = plt.subplots(4,2)
fig3.set_tight_layout(True)
sns.set_style('white')
sns.despine()

ax3[0,0].plot(simtime[-4500:], projections[-4500:,0], color='#443742')
ax3[0,0].set_xlabel('time')
ax3[0,0].set_ylabel('PC1')

ax3[0,1].plot(simtime[-4500:], projections[-4500:,1], color='#443742')
ax3[0,1].set_xlabel('time')
ax3[0,1].set_ylabel('PC2')

ax3[1,0].plot(simtime[-4500:], projections[-4500:,2], color='#443742')
ax3[1,0].set_xlabel('time')
ax3[1,0].set_ylabel('PC3')

ax3[1,1].plot(simtime[-4500:], projections[-4500:,3], color='#443742')
ax3[1,1].set_xlabel('time')
ax3[1,1].set_ylabel('PC4')

ax3[2,0].plot(simtime[-4500:], projections[-4500:,4], color='#443742')
ax3[2,0].set_xlabel('time')
ax3[2,0].set_ylabel('PC5')

ax3[2,1].plot(simtime[-4500:], projections[-4500:,5], color='#443742')
ax3[2,1].set_xlabel('time')
ax3[2,1].set_ylabel('PC6')

ax3[3,0].plot(simtime[-4500:], projections[-4500:,6], color='#443742')
ax3[3,0].set_xlabel('time')
ax3[3,0].set_ylabel('PC7')

ax3[3,1].plot(simtime[-4500:], projections[-4500:,7], color='#443742')
ax3[3,1].set_xlabel('time')
ax3[3,1].set_ylabel('PC8')

fig4, ax4 = plt.subplots()
fig4.set_tight_layout(True)
sns.set_style('white')
sns.despine()
ax4.plot(projections[:,0], projections[:,1], color='#443742')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')

fig5 = plt.figure()
fig5.set_tight_layout(True)
ax5 = fig5.gca(projection='3d')
sns.set_style('white')
sns.despine()
ax5.plot(projections[:,0], projections[:,1], projections[:,8], color='#03adfc')
ax5.set_xlabel('PC1')
ax5.set_ylabel('PC2')
ax5.set_zlabel('PC80')

plt.show()
