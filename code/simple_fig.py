#This file is for producing plots for fig 2 in paper.
from modules.simple_examples import *
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#Globals
linewidth = 3
fontsize = 14
fontweight = 'bold'

if __name__ == '__main__':
    # sns.set_style('white')
    # sns.despine()

    sim, f, train, pred = lorenz(1500, 0.01, 500, 1)

    #For most of the examples
    fig, ax = plt.subplots(1,2)
    fig.set_tight_layout(True)
    ax[0].plot(sim[0], f[0], c='red')
    ax[0].plot(sim[0], train[0], c='blue')
    ax[1].plot(sim[1], pred[0], lw=linewidth, c='red')
    ax[1].plot(sim[1], f[1], lw=linewidth, c='blue')
    fig.legend(['Predicted', 'Target'])
    plt.show()

    fig, ax = plt.subplots(1,2)
    fig.set_tight_layout
    ax[0].plot(simtime[0], train[2][:,0])

    #For A/B/C and maybe K

    #lorenz
    # fig,_ = plt.subplots(1,1)
    # ax = fig.gca(projection='3d')
    #
    # ax.plot(f[0][:,0], f[0][:,1], f[0][:,2])
    # ax.set_xlabel("X Axis")
    # ax.set_ylabel("Y Axis")
    # ax.set_zlabel("Z Axis")
    # ax.set_title("Lorenz Attractor")
    # plt.draw()
    # plt.show()
