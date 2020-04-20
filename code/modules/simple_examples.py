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
        G = np.random.randn(len(simtime), num_waves)
    else:
        G = np.ones((len(simtime), num_waves))

    for i in range(num_waves):
        f += amp[i]*np.sin(np.pi*freq[i]*simtime) + G[:,i]

    return f.reshape((len(simtime),1))

#Fig 2A, 2B, and 2C
#
def sawtooth():
    nsecs = 3000
    dt = 0.1

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    f = sig.sawtooth(simtime*(12*np.pi/nsecs), width=0.5)
    f.reshape((len(simtime), 1))

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime)

    return f, [zt, Wmag], [simtime, zpt]

#Fig 2D
#4 sinusoids
def periodic():
    nsecs = 3000
    dt = 0.1

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    amp = np.array([1,1.5,1,1])
    freq = np.array([1,1,2,3])
    freq = freq*12/nsecs
    f = sinwaves(simtime, 4, amp, freq)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime)

    return f, [zt, Wmag], [simtime, zpt]

#2E
#16 sinusoids
def periodic_cmplx():
    nsecs = 1000
    dt = 0.1

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    amp = np.random.uniform(0, 1, size=16)
    freq = np.random.uniform(0, 1, size=16)
    f = sinwaves(simtime, 16, amp, freq)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime)

    return f, [zt, Wmag], [simtime, zpt]

#2F
#4 sinusoids w/ noise
def noisy():
    nsecs = 1000
    dt = 0.1

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    amp = [1,1.5,1,1]
    freq = [1,1,2,3]
    f = sinwaves(simtime, 4, amp, freq, noise=True)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime)

    return f, [zt, Wmag], [simtime, zpt]

#2G
#square wave
def discont():
    nsecs = 3000
    dt = 0.1

    rnn = Force(g=g_int)

    simtime = np.arange(0, nsecs, dt)
    f = sig.square(simtime/50)
    f.reshape(len(simtime), 1)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime)

    return f, [zt, Wmag], [simtime, zpt]

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
