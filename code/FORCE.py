import numpy as np
from scipy.sparse import random

class Force:
    #Set up network parameters and structure
    #NOTE: Need to implement multiple readouts and inputs
    def __init__(self, *, N=1000, p=0.1, g=1.0, activation=np.tanh, readouts=1):

        self.N = N #Number of neurons in the network
        self.p = p #Sparsity (i.e number of recurrent connections per neuron)
        self.g = g #Chaos in the network, g>1 leads to chaos
        self.activation = activation #Output layer activation
        self.readouts = readouts # Number of readouts

        scale = 1.0/np.sqrt(p*N) #Scale of internal network connections

        #These are the internal network connections
        W_int = random(N, N, density=p, data_rvs=np.random.randn) * g * scale
        self.W_int = W_int.toarray() #Make it a np matrix

        #NOTE: Need to deal with potentially more readouts
        #NOTE: This could be random, but it might not matter as any kinks will
        #get fixed once training starts.
        self.W_out = np.zeros(N) #Readout weights

        #Feedback weights
        #NOTE: Shifts the distribution to mean of zero
        self.W_feed = 2.0*(np.random.rand(N)-0.5)

################################################################################
    #Train the network on specified function
    #NOTE: Need to implement multiple readuts and inputs
    def fit(self, func_to_learn, nsecs, *, alpha=1.0, dt=0.1, learn_every=2):
    #Setting up some stuff
        dW_out = np.zeros(self.N) #Weight update vector

        #Simulation time and length of that vector
        #NOTE: I suppose we are only learning time dependent funcs
        simtime = np.arange(0, nsecs, dt)
        simtime_len = len(simtime)

        ft = func_to_learn(simtime) #Function being learned (vector)

        #Magnitude of weights as we learn
        W_out_mag = np.zeros(simtime_len)

        zt = np.zeros(simtime_len) #Essentially the output function (vector)

    #Okay so now we are leanrning
        #x is pre-activation and z is readout
        #NOTE: Check in to these
        x = 0.5*np.random.randn(self.N)
        z = 0.5*np.random.randn()

        #post-activation
        #NOTE: Could calculate this from parameters
        r = self.activation(x)

        P = (1.0/alpha)*np.eye(self.N) #Inverse correlation matrix

        #Iterate and train the network
        for ti in range(len(simtime)):
            # sim, so x(t) and r(t) are created.
            #NOTE: Check in to this stuff
            x = (1.0-dt)*x + self.W_int.dot(r*dt) + self.W_feed*(z*dt)
            r = self.activation(x)
            z = self.W_out.dot(r)

            if (ti+1) % learn_every == 0:
                #Update inverse correlation matrix
                k = P.dot(r)
                rPr = r.dot(k)
                c = 1.0/(1.0 + rPr)
                P = P - np.outer(k, k * c)

                #Update the error for the linear readout
                e = z - ft[ti]

                #Update the output weights
                dW_out = -e * k * c
                self.W_out = self.W_out + dW_out

            #Store the output of the system.
            zt[ti] = z

            #Magnitude of weights
            W_out_mag[ti] = np.linalg.norm(self.W_out)

        #Average error after learning
        error_avg = np.sum(np.abs(zt-ft))/simtime_len
        print('Training MAE: {:.3f}'.format(error_avg))

        #Return the training progression
        return zt, W_out_mag

################################################################################
    #Use the trained neural network predict or generate
    #NOTE: Need to consider multiple readouts and inputs
    def predict(self, x, start, end, dt):
        simtime = np.arange(start, end, dt)
        simtime_len = len(simtime)

        zpt = np.zeros(simtime_len)

        r = self.activation(x)
        z = self.W_out.dot(r)

        for ti in range(len(simtime)):

            # sim, so x(t) and r(t) are created.
            x = (1.0-dt)*x + self.W_int.dot(r*dt) + self.W_feed*(z*dt)
            r = self.activation(x)
            z = self.W_out.dot(r)

            zpt[ti] = z

        return simtime, zpt

################################################################################
    #Evaluate the neural network
    #NOTE: Need to consider multiple readouts and inputs
    #NOTE: Should check on all of this stuff
    def evaluate(self, start, end, dt, func_learned):

        x = 0.5*np.random.randn(self.N)
        t, zpt = self.predict(x, start, end, dt)

        simtime_len = len(zpt)

        ft = func_learned(t)

        error_avg = np.sum(np.abs(zpt-ft))/simtime_len
        print('Testing MAE: {:.3f}'.format(error_avg))

        return error_avg
