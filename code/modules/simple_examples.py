#This file is for recreating fig 2 from paper
#Includes all simple examples
from .FORCE import Force
import numpy as np
import scipy.signal as sig

#Globals
N = 1000
int_p = 0.1
out_p = 1
g_int = 1.5
g_out = 1
g_feed = 0
alpha = 1.0
Ni = 0

#Function for producing sum of sine waves
def sinwaves(simtime, num_waves, amp=[1.0], freq=[1/np.pi], noise=False):
    f = np.zeros(len(simtime))

    if noise:
        G = np.random.randn(len(simtime), num_waves)/10.0
    else:
        G = np.ones((len(simtime), num_waves))

    for i in range(num_waves):
        f += amp[i]*np.sin(np.pi*freq[i]*simtime) + G[:,i]

    return f.reshape((len(simtime),1))

#Basic sinwave
def basic(nsecs, dt):
    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    amp = np.array([1])
    freq = np.array([1])
    freq = freq*12/nsecs
    f = sinwaves(simtime, 1, amp, freq)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime)

    return f, [zt, Wmag], [simtime, zpt]

#Fig 2A, 2B, and 2C
#
def triangle(nsecs, dt):

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    simtime2 = np.arange(nsecs, nsecs*2, dt)

    f = sig.sawtooth(simtime*(10.0*np.pi/nsecs), width=0.5)
    f.reshape((len(simtime), 1))
    f2 = sig.sawtooth(simtime*(10.0*np.pi/nsecs), width=0.5)
    f2.reshape((len(simtime), 1))

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [f, f2] , [zt, Wmag], [simtime, zpt]

#Fig 2D
#4 sinusoids
def periodic(nsecs, dt):

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    simtime2 = np.arange(nsecs, nsecs*2, dt)

    amp = np.array([1.0, 1.0/2.0, 1.0/3.0, 1.0/6.0]) * 1.3
    freq = np.array([1.0, 2.0, 5.0, 6.0]) * 10.0 / nsecs
    f = sinwaves(simtime, 4, amp, freq)/1.5
    f2 = sinwaves(simtime2, 4, amp, freq)/1.5

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [f, f2], [zt, Wmag], [simtime, zpt]

#2E
#16 sinusoids
def periodic_cmplx(nsecs, dt):

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    simtime2 = np.arange(nsecs, nsecs*2, dt)

    amp = np.array([1.0, 1.0/2.0, 1.0/3.0, 1.0/6.0,
                    1.0/2.5, 1.0/7.0, 1.0/3.2, 1.0/4.0,
                    1.0/1.5, 1.0/5.0, 1.0/1.2, 1.0/3.4,
                    1.0/4.7, 1.0/6.9, 1.0/10.0, 1.0/3.8]) * 1.3
    freq = np.array([1.0, 2.0, 3.0, 6.0,
                    2.5, 7.0, 3.2, 4.0,
                    1.5, 5.0, 1.2, 3.4,
                    4.7, 6.9, 10.0, 3.8]) * 10.0 / nsecs

    f = sinwaves(simtime, 16, amp, freq)/1.5
    f2 = sinwaves(simtime2, 16, amp, freq)/1.5

    # zt, Wmag, x = rnn.fit(simtime, f)
    # zpt = rnn.predict(x, simtime2)
    zt = 0
    Wmag = 0
    x = 0
    zpt = 0

    return [f, f2], [zt, Wmag], [simtime, zpt]

#2F
#4 sinusoids w/ noise
def noisy(nsecs, dt):

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    simtime2 = np.arange(nsecs, nsecs*2, dt)

    amp = np.array([1.0, 1.0/2.0, 1.0/3.0, 1.0/6.0])*1.3
    freq = np.array([1.0, 2.0, 3.0, 4.0])*10.0/nsecs
    f = sinwaves(simtime, 4, amp, freq, noise=True)/1.5
    f2 = sinwaves(simtime2, 4, amp, freq, noise=True)/1.5

    # zt, Wmag, x = rnn.fit(simtime, f)
    # zpt = rnn.predict(x, simtime2)
    zt = 0
    Wmag = 0
    x = 0
    zpt = 0

    return [f, f2], [zt, Wmag], [simtime, zpt]

#2G
#square wave
def discont(nsecs, dt):

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    simtime2 = np.arange(nsecs, nsecs*2, dt)

    f = sig.square(simtime/50)
    f.reshape(len(simtime), 1)
    f2 = sig.square(simtime2/50)
    f2.reshape(len(simtime2), 1)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [f, f2], [zt, Wmag], [simtime, zpt]

################################################################################
#These will need some more work

#2H
#Lorenz attractor 1D slice
def lorenz():
    rnn = Force(g=g_int)

#2I
#sine waves with period 60ms and 8s
def sin_multiT():
    rnn = Force(g=g_int)

#2J
#One shot example with two outputs
def aperiodic():
    rnn = Force(g=g_int, readouts=2)

#2K
#FORCE failing
def fail():
    rnn = Force(g=g_int)
