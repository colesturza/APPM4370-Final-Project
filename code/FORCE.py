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
        self.W_out = np.zeros((N, readouts)) #Readout weights

        #Feedback weights
        #NOTE: Shifts the distribution to mean of zero
        self.W_feed = 2.0*(np.random.rand(N, readouts)-0.5)

################################################################################
    #Train the network on specified function
    #NOTE: Need to implement multiple readuts and inputs
    def fit(self, simtime, func_to_learn, *, alpha=1.0, learn_every=2):

    #Setting up some stuff

        #NOTE: I suppose we are only learning time dependent funcs
        #Simulation time and length of that vector
        simtime_len = simtime.shape[0]

        diff = np.diff(simtime)
        if not np.any(np.isclose(diff, diff[0])):
            raise ValueError('All values in simtime must be evenly spaced.')

        dt = diff[0]

        #Weight update vector
        dW_out = np.zeros((self.N, self.readouts))

        #Check if func_to_learn is either a callable function or an
        #ndarray of values to learn.
        if callable(func_to_learn):
            ft = func_to_learn(simtime) #Function being learned (vector)
        elif type(func_to_learn) is np.ndarray:
            ft = func_to_learn #Input is an array
        else:
            raise ValueError("""func_to_learn must either be a callable function
                or a numpy ndarray of shape (n, {}).""".format(self.readouts))

        ft = ft.reshape((simtime_len, self.readouts))

        #Magnitude of weights as we learn
        W_out_mag = np.zeros((simtime_len, self.readouts))

        #Essentially the output function (vector)
        zt = np.zeros((simtime_len, self.readouts))

    #Okay so now we are leanrning

        #x is pre-activation and z is readout
        #NOTE: Check in to these
        x = 0.5*np.random.randn(self.N, 1)
        z = 0.5*np.random.randn(self.readouts, 1)

        #post-activation
        #NOTE: Could calculate this from parameters
        r = self.activation(x)

        P = (1.0/alpha)*np.eye(self.N) #Inverse correlation matrix

        #Iterate and train the network
        for ti in range(simtime_len):

            # sim, so x(t) and r(t) are created.
            #NOTE: Check in to this stuff
            x = (1.0-dt)*x + self.W_int.dot(r*dt) + self.W_feed.dot(z)*dt
            r = self.activation(x)
            z = self.W_out.T.dot(r)

            if (ti+1) % learn_every == 0:
                #Update inverse correlation matrix
                k = P.dot(r)
                rPr = r.T.dot(k)
                c = 1.0/(1.0 + rPr)
                P = P - np.outer(k, k * c)

                #Update the error for the linear readout
                e = z - ft[ti].reshape((self.readouts, 1))

                #Update the output weights
                dW_out = -k.dot(e.T) * c
                self.W_out = self.W_out + dW_out

            #Store the output of the system.
            zt[ti,:] = z.reshape(self.readouts)

            #Magnitude of weights
            W_out_mag[ti,:] = np.linalg.norm(self.W_out, axis=0)

        #Average error after learning
        error_avg = np.sum(np.abs(np.subtract(zt, ft)))/simtime_len
        print('Training MAE: {:.5f}'.format(error_avg))

        #Return the training progression
        return zt, W_out_mag, x

################################################################################
    #Use the trained neural network predict or generate
    #NOTE: Need to consider multiple readouts and inputs
    def predict(self, x, simtime):

        simtime_len = simtime.shape[0]

        diff = np.diff(simtime)
        if not np.any(np.isclose(diff, diff[0])):
            raise ValueError('All values in simtime must be evenly spaced.')

        dt = diff[0]

        zpt = np.zeros((simtime_len, self.readouts))

        r = self.activation(x)
        z = self.W_out.T.dot(r)

        for ti in range(simtime_len):

            # sim, so x(t) and r(t) are created.
            x = (1.0-dt)*x + self.W_int.dot(r*dt) + self.W_feed.dot(z)*dt
            r = self.activation(x)
            z = self.W_out.T.dot(r)

            zpt[ti,:] = z.reshape(self.readouts)

        return zpt

################################################################################
    #Evaluate the neural network
    #NOTE: Need to consider multiple readouts and inputs
    #NOTE: Should check on all of this stuff
    def evaluate(self, x, simtime, func_learned):

        zpt = self.predict(x, simtime)

        #NOTE: I suppose we are only learning time dependent funcs
        #Simulation time and length of that vector
        simtime_len = simtime.shape[0]

        #Check if func_to_learn is either a callable function or an
        #ndarray of values to learn.
        if callable(func_learned):
            ft = func_learned(simtime) #Function being learned (vector)
        elif type(func_learned) is np.ndarray:
            ft = func_learned #Input is an array
        else:
            raise ValueError("""func_learned must either be a callable function
                or a numpy ndarray of shape (n, {}).""".format(self.readouts))

        ft = ft.reshape((simtime_len, self.readouts))

        error_avg = np.sum(np.abs(np.subtract(zpt, ft)))/simtime_len
        print('Testing MAE: {:.5f}'.format(error_avg))

        return error_avg
