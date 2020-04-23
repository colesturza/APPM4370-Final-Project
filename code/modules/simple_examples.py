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
        avgA = sum(amp)/len(amp)
        G = np.random.randn(len(simtime), num_waves)*avgA/4.0
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
    rnn.config(neuron_output=True)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    freq = 1/60
    f = sig.sawtooth(simtime*np.pi*freq, width=0.5)
    f.reshape((len(simtime),1))

    f2 = sig.sawtooth(simtime2*np.pi*freq, width=0.5)
    f2.reshape((len(simtime2),1))

    zt, Wmag = rnn.fit(simtime, f)
    intOutT = rnn.intOut
    zpt = rnn.predict(simtime2)
    intOutP = rnn.intOut

    return [simtime, simtime2], [f, f2] , [zt, Wmag, intOutT], [zpt, intOutP]

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

    zt, Wmag = rnn.fit(simtime, f)
    zpt = rnn.predict(simtime2)

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

    zt, Wmag = rnn.fit(simtime, f)
    zpt = rnn.predict(simtime2)

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

    zt, Wmag = rnn.fit(simtime, f)
    zpt = rnn.predict(simtime2)

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

    zt, Wmag = rnn.fit(simtime, f)
    zpt = rnn.predict(simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

############################################################################

#2I
#sine wave with period 800 T or 6T
def sin(Ttime, dt, Ptime, vars):
    A, F = vars
    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    amp = 1*A
    freq = (2/F)*np.pi

    f = sinwaves(simtime, 1, [amp], [freq])
    f2 = sinwaves(simtime2, 1, [amp], [freq])

    zt, Wmag = rnn.fit(simtime, f)
    zpt = rnn.predict(simtime2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

################################################################################
a, r, b = 10, 28, 8/3

def system(S):
    x, y, z = S
    return np.array([a * (y - x), x * (r - z) - y, x * y - b * z])

def fwdEuler(dimensions, t0, tf, V0, dt):
    t = np.arange(t0,tf,dt)
    V = np.zeros((len(t), 3))
    V[0] = V0

    #FWD Euler
    for i in range(len(t)-1):
        Vprime = system(V[i])
        V[i+1] = V[i] + dt*Vprime

    return t, V[:,0:dimensions]

#2H
#Lorenz attractor 1D slice
def lorenz(Ttime, dt, Ptime, dims=1):
    rnn = Force(g=g_int, readouts=dims)

    V0 = np.array([0, 1, 2])
    t, V = fwdEuler(dims, 0, Ttime, V0, dt)
    t2, V2 = fwdEuler(dims, Ttime, Ttime+Ptime, V[-1], dt)

    zt, Wmag = rnn.fit(t, V[:,:dims])
    zpt = rnn.predict(t2)

    return [t, t2], [V[:,:dims], V2[:,:dims]] , [zt, Wmag], zpt

################################################################################

#2J
#One shot example with two outputs
def aperiodic(Ttime, dt, Ptime):
    rnn = Force(g=g_int, readouts=2)

    return [simtime, simtime2], [f, f2] , [zt, Wmag], zpt

################################################################################

#2K
#Failing due to low amplitude
def low_amp(Ttime, dt, Ptime, vars):
    A, F = vars

    rnn = Force(g=g_int)

    simtime = np.arange(0, Ttime, dt)
    simtime2 = np.arange(Ttime, Ttime+Ptime, dt)

    amp = 1*A
    freq = (2/F)*np.pi

    f = sinwaves(simtime, 1, [amp], [freq])
    f2 = sinwaves(simtime2, 1, [amp], [freq])

    zt, Wmag = rnn.fit(simtime, f)
    intOutT = rnn.intOut
    zpt = rnn.predict(simtime2)
    intOutP = rnn.intOut

    return [simtime, simtime2], [f, f2] , [zt, Wmag, intOutT], [zpt, intOutP]
