#This file is for producing plots for fig 2 in paper.
from modules.simple_examples import *
import matplotlib.pyplot as plt
import seaborn as sns

#Globals
linewidth = 3
fontsize = 14
fontweight = 'bold'

if __name__ == '__main__':
    sns.set_style('white')
    sns.despine()

    sim, f, train, pred = sin_8s(1500, 0.1, 500)

    #For most of the examples
    fig, ax = plt.subplots(1,2)
    fig.set_tight_layout(True)
    ax[0].plot(sim[0], f[0], c='red')
    ax[0].plot(sim[0], train[0], c='blue')
    ax[1].plot(sim[1], pred, lw=linewidth, c='red')
    ax[1].plot(sim[1], f[1], lw=linewidth, c='blue')
    fig.legend(['Predicted', 'Target'])
    plt.show()

    #For A/B/C and maybe K
