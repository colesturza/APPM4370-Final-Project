#This file is for producing plots for fig 5 in paper.
from modules.FORCE import Force
from modules.simple_examples import sinwaves
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Globals
linewidth = 3
fontsize = 14
fontweight = 'bold'

if __name__ == '__main__':
    # sns.set_style('white')
    # sns.despine()

    Ttime = 1500
    Ptime = 500
    dt = 0.1
    g_ints = np.linspace(0.75, 1.56, 18)

    rmss = []
    Wmags = []

    for g_int in g_ints:

        rnn = Force(g=g_int)

        simtime = np.arange(0, Ttime, dt)
        simtime2 = np.arange(Ttime, Ttime+Ptime, dt)
        simtime2_len = len(simtime2)

        amp = np.array([1, 1/2, 1/3, 1/6])
        freq = np.array([1, 2, 3, 4])*np.pi*(1/60)
        f = sinwaves(simtime, 4, amp, freq)
        f2 = sinwaves(simtime2, 4, amp, freq)

        zt, W_out_t = rnn.fit(simtime, f)
        zpt = rnn.predict(simtime2)

        rms = np.sqrt(np.sum(np.subtract(zpt, f2)**2)/simtime2_len)
        Wmag = np.linalg.norm(W_out_t)

        rmss.append(rms)
        Wmags.append(Wmag)

    #For most of the examples
    fig, ax = plt.subplots(2,1)
    fig.set_tight_layout(True)
    ax[0].plot(g_ints, rmss, c='k', marker='s', mfc='g', mec='k')
    ax[0].set_xlabel('g')
    ax[0].set_ylabel('RMS error')
    ax[1].plot(g_ints, Wmags, c='k', marker='s', mfc='g', mec='k')
    ax[1].set_xlabel('g')
    ax[1].set_ylabel('|w|')
    plt.show()
