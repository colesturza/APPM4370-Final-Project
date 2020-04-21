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

    f, train, pred = noisy(1500, 0.01)

    fig, ax = plt.subplots(1,1)
    fig.set_tight_layout(True)
    # ax.plot(pred[0], pred[1], lw=linewidth, c='red')
    ax.plot(pred[0], f[1], lw=linewidth, c='blue')
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
