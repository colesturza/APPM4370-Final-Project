#This file is for producing plots for fig 2 in paper.
from modules.simple_examples import *
import matplotlib.pyplot as plt
import seaborn as sns

#Globals
linewidth = 3
fontsize = 14
fontweight = 'bold'

if __name__ == '__main__':
    sns.set()

    sim, f, train, pred = triangle(200, 0.005, 3)

    fig, ax = plt.subplots(1,2)
    fig.set_tight_layout(True)
    ax[0].plot(sim[0], train[0])
    ax[0].plot(sim[0], f[0])
    ax[1].plot(sim[1], pred, lw=linewidth, c='red')
    ax[1].plot(sim[1], f[1], lw=linewidth, c='blue')
    plt.show()

    # nsecs = 3000
    # dt = 0.1
    #
    # simtime = np.arange(0, nsecs, dt)
    # f = sig.sawtooth(simtime*(12*np.pi/nsecs), width=0.5)
    # f.reshape((len(simtime), 1))
    #
    # fig, ax = plt.subplots(1,1)
    # ax.plot(simtime, f)
    # plt.show()
