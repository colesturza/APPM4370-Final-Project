from modules.PCA import PCA_NN
from modules.FORCE import Force
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

linewidth = 3
fontsize = 14
fontweight = 'bold'

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

# rnn = PCA_NN(N=N, p=p, g=g)

# for i in range(2, 4):
#     zt, x, eigvals, projections = rnn.fit(simtime, sum_of_four_sinusoids, alpha=alpha, learn_every=learn_every)
#     projections = projections.reshape((simtime_len, 9))
#     df = pd.DataFrame(data=projections)
#     df.to_csv('projections{}'.format(i), index=False)

#
# zpt = rnn.predict(x, simtime2)
# avg_error = rnn.evaluate(x, simtime2, sum_of_four_sinusoids)

# rnn = Force(N=N, p=p, g=g)
# zt, _ = rnn.fit(simtime, sum_of_four_sinusoids, alpha=alpha, learn_every=learn_every)
#
# fig1, axs = plt.subplots(2,1)
# fig1.set_tight_layout(True)
# sns.set_style('white')
# sns.despine()
#
# axs[0].plot(simtime[-4500:], sum_of_four_sinusoids(simtime)[-4500:], color='firebrick', lw=linewidth)
# axs[0].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
# axs[0].set_ylabel('f', fontsize=fontsize, fontweight=fontweight)
# axs[0].set_title('Actual', fontsize=fontsize, fontweight=fontweight)
# axs[0].set_yticklabels([])
# axs[0].set_xticklabels([])
#
# axs[1].plot(simtime[-4500:], zt[-4500:], color='#aba078', lw=linewidth)
# axs[1].set_xlabel('time', fontsize=fontsize, fontweight=fontweight)
# axs[1].set_ylabel('z', fontsize=fontsize, fontweight=fontweight)
# axs[1].set_title('Training', fontsize=fontsize, fontweight=fontweight)
# axs[1].set_yticklabels([])
# axs[1].set_xticklabels([])
#
# eigvals = pd.read_csv('eigvals.txt', header=None).to_numpy()
projections1 = pd.read_csv('projections1').to_numpy()
projections2 = pd.read_csv('projections2').to_numpy()
projections3 = pd.read_csv('projections3').to_numpy()
#
# fig2, ax2 = plt.subplots()
# fig2.set_tight_layout(True)
# sns.set_style('white')
# sns.despine()
# ax2.plot(np.arange(1,101), np.log10(eigvals), color='slateblue', lw=linewidth)
# ax2.set_xlabel('eigenvalue', fontsize=fontsize, fontweight=fontweight)
# ax2.set_ylabel('log10(eigenvalue)', fontsize=fontsize, fontweight=fontweight)
#
# # set the x-spine
# ax2.spines['left'].set_position('zero')
#
# # turn off the right spine/ticks
# ax2.spines['right'].set_color('none')
# ax2.yaxis.tick_left()
#
# # set the y-spine
# ax2.spines['bottom'].set_position('zero')
#
# # turn off the top spine/ticks
# ax2.spines['top'].set_color('none')
# ax2.xaxis.tick_bottom()
#
# fig3, ax3 = plt.subplots(4,2)
# fig3.set_tight_layout(True)
# sns.set_style('white')
# sns.despine()
#
# ax3[0,0].plot(simtime[-4500:], projections1[-4500:,0], color='#aba078')
# ax3[0,0].set_xlabel('time')
# ax3[0,0].set_ylabel('PC1')
# ax3[0,0].set_yticklabels([])
# ax3[0,0].set_xticklabels([])
#
# ax3[0,1].plot(simtime[-4500:], projections1[-4500:,1], color='#aba078')
# ax3[0,1].set_xlabel('time')
# ax3[0,1].set_ylabel('PC2')
# ax3[0,1].set_yticklabels([])
# ax3[0,1].set_xticklabels([])
#
# ax3[1,0].plot(simtime[-4500:], projections1[-4500:,2], color='#aba078')
# ax3[1,0].set_xlabel('time')
# ax3[1,0].set_ylabel('PC3')
# ax3[1,0].set_yticklabels([])
# ax3[1,0].set_xticklabels([])
#
# ax3[1,1].plot(simtime[-4500:], projections1[-4500:,3], color='#aba078')
# ax3[1,1].set_xlabel('time')
# ax3[1,1].set_ylabel('PC4')
# ax3[1,1].set_yticklabels([])
# ax3[1,1].set_xticklabels([])
#
# ax3[2,0].plot(simtime[-4500:], projections1[-4500:,4], color='#aba078')
# ax3[2,0].set_xlabel('time')
# ax3[2,0].set_ylabel('PC5')
# ax3[2,0].set_yticklabels([])
# ax3[2,0].set_xticklabels([])
#
# ax3[2,1].plot(simtime[-4500:], projections1[-4500:,5], color='#aba078')
# ax3[2,1].set_xlabel('time')
# ax3[2,1].set_ylabel('PC6')
# ax3[2,1].set_yticklabels([])
# ax3[2,1].set_xticklabels([])
#
# ax3[3,0].plot(simtime[-4500:], projections1[-4500:,6], color='#aba078')
# ax3[3,0].set_xlabel('time')
# ax3[3,0].set_ylabel('PC7')
# ax3[3,0].set_yticklabels([])
# ax3[3,0].set_xticklabels([])
#
# ax3[3,1].plot(simtime[-4500:], projections1[-4500:,7], color='#aba078')
# ax3[3,1].set_xlabel('time')
# ax3[3,1].set_ylabel('PC8')
# ax3[3,1].set_yticklabels([])
# ax3[3,1].set_xticklabels([])

fig4, ax4 = plt.subplots()
fig4.set_tight_layout(True)
sns.set_style('white')
sns.despine()
ax4.plot(projections1[:,0][-4000:], projections1[:,1][-4000:], color='#aba078')
ax4.plot(projections2[:,0][-4000:], projections2[:,1][-4000:], color='#78725c')
ax4.plot(projections3[:,0][-4000:], projections3[:,1][-4000:], color='#423f33')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')

fig5 = plt.figure()
fig5.set_tight_layout(True)
ax5 = fig5.gca(projection='3d')
sns.set_style('white')
sns.despine()
ax5.plot(projections1[:,0][2::50], projections1[:,1][2::50], projections1[:,8][2::50], color='#aba078')
ax5.set_xlabel('PC1')
ax5.set_ylabel('PC2')
ax5.set_zlabel('PC80')

plt.show()
