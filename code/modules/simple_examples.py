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

############################################################################

#Function for producing sum of sine waves
def sinwaves(simtime, num_waves, amp, freq, noise=False):
    f = np.zeros(len(simtime))

    if noise:
        G = np.random.randn(len(simtime), num_waves)/4.0
    else:
        G = np.zeros((len(simtime), num_waves))

    for i in range(num_waves):
        f += amp[i]*np.sin(freq[i]*simtime) + G[:,i]

    return f.reshape((len(simtime),1))

############################################################################

#Fig 2A, 2B, and 2C
#Run for 3
def triangle(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    freq = 1/60
    f = sig.sawtooth(simtime*np.pi*freq, width=0.5)
    f.reshape((len(simtime),1))

    f2 = sig.sawtooth(simtime2*np.pi*freq, width=0.5)
    f2.reshape((len(simtime2),1))

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

############################################################################

#Fig 2D
#4 sinusoids
#Run for 3
def periodic(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    amp = np.array([1, 1/2, 1/3, 1/6])
    freq = np.array([1, 2, 3, 4])*np.pi*(1/60)
    f = sinwaves(simtime, 4, amp, freq)
    f2 = sinwaves(simtime2, 4, amp, freq)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

############################################################################

#2E
#16 sinusoids
#Run for 5
def periodic_cmplx(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    freq = 1/60
    amp = np.array([1, 1/2, 1/6, 1/3,
                    1/8, 1/5, 1/6, 1/10,
                    1/12, 1/4, 1/8, 1/2,
                    1/8, 1/5, 1/6, 1/10])
    freq = np.array([1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 10, 11, 12,
                    13, 14, 15, 16])*np.pi*freq

    f = sinwaves(simtime, 16, amp, freq)
    f2 = sinwaves(simtime2, 16, amp, freq)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

############################################################################

#2F
#4 sinusoids w/ noise
def noisy(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    amp = np.array([1, 1/2, 1/3, 1/6])
    freq = np.array([1, 2, 3, 4])*np.pi*(1/60)
    f = sinwaves(simtime, 4, amp, freq, noise=True)
    f2 = sinwaves(simtime2, 4, amp, freq, noise=True)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

############################################################################

#2G
#square wave
def discont(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    freq = 1/60
    f = sig.square(simtime*np.pi*freq)
    f.reshape(len(simtime), 1)
    f2 = sig.square(simtime2*np.pi*freq)
    f2.reshape(len(simtime2), 1)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

############################################################################

#2I
#sine wave with period 8s
def sin_8s(Ttime, dt, Ptime):
    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    amp = 1
    freq = (4.0/16.0)*np.pi

    f = sinwaves(simtime, 1, [amp], [freq])
    f2 = sinwaves(simtime2, 1, [amp], [freq])

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

############################################################################

#2I
#sine wave with period 60ms
def sin_60ms(Ttime, dt, Ptime):
    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    amp = 3
    freq = 2000.0/60.0*np.pi

    f = sinwaves(simtime, 1, [amp], [freq])
    f2 = sinwaves(simtime2, 1, [amp], [freq])

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

################################################################################
#These will need some more work

#2H
#Lorenz attractor 1D slice
def lorenz(Ttime, dt, Ptime):
    rnn = Force(g=g_int)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

#2J
#One shot example with two outputs
def aperiodic(Ttime, dt, Ptime):
    rnn = Force(g=g_int, readouts=2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

#2K
#FORCE failing
def fail(Ttime, dt, Ptime):
    rnn = Force(g=g_int)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt
