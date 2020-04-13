import numpy as np
from scipy.sparse import random

class Force:

    def __init__(self, *, N=1000, p=0.1, g=1.0, activation=np.tanh, readouts=1):

        self.N = N #Number of neurons in the network
        self.p = p #Sparsity (i.e number of recurrent connections per neuron)
        self.g = g #Chaos in the network, g>1 leads to chaos
        self.activation = activation #Output layer activation

        scale = 1.0/np.sqrt(p*N) #Scale of internal network connections

        #These are the internal network connections
        W_int = random(N, N, density=p, data_rvs=np.random.randn) * g * scale
        self.W_int = W_int.toarray() #Make it a np matrix

        #NOTE: Need to deal with potentially more readouts
        #NOTE: Shouldnt this be random
        self.W_out = np.zeros(N) #Readout weights

        #Not sure whats going on here
        self.wf = 2.0*(np.random.rand(N)-0.5)

        self.x = 0.5*np.random.randn(N)
        self.z = 0.5*np.random.randn()

    def fit(self, func_to_learn, nsecs, *, alpha=1.0, dt=0.1, learn_every=2):

        #Setting up some stuff

        dW_out = np.zeros(self.N) #Weight update vector

        #Simulation time and length of that vector
        #NOTE: What if we aren't learning time dependent function
        simtime = np.arange(0, nsecs, dt)
        simtime_len = len(simtime)

        ft = func_to_learn(simtime) #Function being learned (vector)

        #Not sure what this is
        #NOTE: I think this vector of ouput weights or something
        wo_len = np.zeros(simtime_len)

        zt = np.zeros(simtime_len) #Essentially the output function (vector)

        #Okay so now we are leanrning

        #Not sure about this
        #NOTE: Why are we calculating this here if we recalculate a few lines down
        r = self.activation(self.x)

        P = (1.0/alpha)*np.eye(self.N) #Inverse correlation matrix

        for ti, t in enumerate(simtime):
            # sim, so x(t) and r(t) are created.
            self.x = (1.0-dt)*self.x + self.W_int.dot(r*dt) + self.wf*(self.z*dt)
            r = self.activation(self.x)
            self.z = self.W_out.dot(r)

            if (ti+1) % learn_every == 0:
                # update inverse correlation matrix
                k = P.dot(r)
                rPr = r.dot(k)
                c = 1.0/(1.0 + rPr)
                P = P - np.outer(k, k * c)

                # update the error for the linear readout
                e = self.z - ft[ti]

                # update the output weights
                dW_out = -e * k * c

                self.W_out = self.W_out + dW_out

            # Store the output of the system.
            zt[ti] = self.z
            wo_len[ti] = np.sqrt(self.W_out.dot(self.W_out))

        error_avg = np.sum(np.abs(zt-ft))/simtime_len
        print('Training MAE: {:.3f}'.format(error_avg))

        return zt, wo_len


    def predict(self, start, end, dt):

        simtime = np.arange(start, end, dt)
        simtime_len = len(simtime)

        zpt = np.zeros(simtime_len)

        x = self.x
        r = self.activation(x)
        z = self.W_out.dot(r)

        for ti, t in enumerate(simtime):

            # sim, so x(t) and r(t) are created.
            x = (1.0-dt)*x + self.W_int.dot(r*dt) + self.wf*(z*dt)
            r = self.activation(x)
            z = self.W_out.dot(r)

            zpt[ti] = z

        return t, zpt

    def evaluate(self, start, end, dt, func_learned):

        t, zpt = self.predict(start, end, dt)

        simtime_len = len(zpt)

        ft = func_to_learn(t)

        error_avg = np.sum(np.abs(zpt-ft))/simtime_len
        print('Testing MAE: {:.3f}'.format(error_avg))

        return error_avg
