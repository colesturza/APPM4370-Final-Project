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
        G = np.random.randn(len(simtime), num_waves)/20.0
    else:
        G = np.ones((len(simtime), num_waves))

    for i in range(num_waves):
        f += amp[i]*np.sin(np.pi*freq[i]*simtime) + G[:,i]

    return f.reshape((len(simtime),1))

#Basic sinwave
# def basic(Ttime, dt, Ptime):
#     rnn = Force(g=g_int)
#
#     simtime = np.arange(0, Ttime, dt)
#     amp = np.array([1])
#     freq = np.array([1])
#     freq = freq*12/nsecs
#     f = sinwaves(simtime, 1, amp, freq)
#
#     zt, Wmag, x = rnn.fit(simtime, f)
#     zpt = rnn.predict(x, simtime)
#
#     return f, [zt, Wmag], [simtime, zpt]

#Fig 2A, 2B, and 2C
#Run for 3
def triangle(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt/10)

    f = sig.sawtooth(simtime*(2*np.pi), width=0.5)
    f.reshape((len(simtime), 1))
    f2 = sig.sawtooth(simtime2*(2*np.pi), width=0.5)
    f2.reshape((len(simtime2), 1))

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

#Fig 2D
#4 sinusoids
#Run for 3
def periodic(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt/1000)

    amp = np.array([1.0, 1.0/2.0, 1.0/3.0, 1.0/6.0]) * 1.3
    freq = np.array([1.0, 2.0, 3.0, 4.0])
    f = sinwaves(simtime, 4, amp, freq)/1.5
    f2 = sinwaves(simtime2, 4, amp, freq)/1.5

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], [zpt]

#2E
#16 sinusoids
#Run for 5
def periodic_cmplx(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt/1000)

    amp = np.array([1.0, 1.0/8.0, 1.0/8.0, 1.0/8.0,
                    1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0,
                    1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0,
                    1.0/8.0, 1.0/8.0, 1.0/8.0, 1.0/8.0]) * 1.3
    freq = np.array([2.0, 3.0, 3.5, 4.0,
                    4.5, 5.0, 5.5, 6.0,
                    6.5, 7.0, 7.5, 8.0,
                    8.5, 9.0, 9.5, 10.0])

    f = sinwaves(simtime, 16, amp, freq)/1.5
    f2 = sinwaves(simtime2, 16, amp, freq)/1.5

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], [zpt]

#2F
#4 sinusoids w/ noise
def noisy(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt/1000)

    amp = np.array([1.0, 1.0/2.0, 1.0/3.0, 1.0/6.0])*1.3
    freq = np.array([1.0, 2.0, 3.0, 4.0])
    f = sinwaves(simtime, 4, amp, freq, noise=True)/1.5
    f2 = sinwaves(simtime2, 4, amp, freq, noise=True)/1.5

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], [zpt]

#2G
#square wave
def discont(Ttime, dt, Ptime):

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    f = sig.square(simtime*2*np.pi)
    f.reshape(len(simtime), 1)
    f2 = sig.square(simtime2*2*np.pi)
    f2.reshape(len(simtime2), 1)

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], [zpt]

#2I
#sine wave with period 8s
def sin_8s(Ttime, dt, Ptime):
    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt/1000)

    amp = 1
    freq = 4.0/16.0

    f = sinwaves(simtime, 1, [amp], [freq])
    f2 = sinwaves(simtime2, 1, [amp], [freq])

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], [zpt]

#2I
#sine wave with period 60ms
def sin_60ms(Ttime, dt, Ptime):
    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt/1000)

    amp = 3
    freq = 2000.0/60.0

    f = sinwaves(simtime, 1, [amp], [freq])
    f2 = sinwaves(simtime2, 1, [amp], [freq])

    zt, Wmag, x = rnn.fit(simtime, f)
    zpt = rnn.predict(x, simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], [zpt]

################################################################################
#These will need some more work

#2H
#Lorenz attractor 1D slice
def lorenz(Ttime, dt, Ptime):
    rnn = Force(g=g_int)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], [zpt]

#2J
#One shot example with two outputs
def aperiodic(Ttime, dt, Ptime):
    rnn = Force(g=g_int, readouts=2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], [zpt]

#2K
#FORCE failing
def fail(Ttime, dt, Ptime):
    rnn = Force(g=g_int)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], [zpt]
