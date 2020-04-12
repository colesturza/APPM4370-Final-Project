import numpy as np
from scipy.sparse import random

class force:

    def __init__(self, N=1000, p=0.1, g=1.0, activation=np.tanh):

        self.N = N
        self.p = p
        self.g = g
        self.activation = activation

        scale = 1.0/np.sqrt(p*N)
        M = random(N, N, density=p, data_rvs=np.random.randn) * g * scale

        self.M = M.toarray()

        self.wo = np.zeros(N)
        self.wf = 2.0*(np.random.rand(N)-0.5)

        self.x = 0.5*np.random.randn(N)
        self.z = 0.5*np.random.randn()

    def fit(self, alpha=1.0, nsecs, dt=0.1, learn_every=2, func_to_learn):

        dw = np.zeros(self.N)

        simtime = np.arange(0, nsecs, dt)
        simtime_len = len(simtime)

        ft = func_to_learn(simtime)

        wo_len = np.zeros(simtime_len)
        zt = np.zeros(simtime_len)

        r = np.activation(self.x)

        P = (1.0/alpha)*np.eye(self.N)

        for ti, t in enumerate(simtime):

            # sim, so x(t) and r(t) are created.
            self.x = (1.0-dt)*x + self.M.dot(r*dt) + self.wf*(self.z*dt)
            r = np.activation(self.x)
            self.z = self.wo.dot(r)

            if (ti+1) % learn_every == 0:
                # update inverse correlation matrix
                k = P.dot(r)
                rPr = r.dot(k)
                c = 1.0/(1.0 + rPr)
                P = P - np.outer(k, k * c)

                # update the error for the linear readout
                e = self.z - ft[ti]

                # update the output weights
                dw = -e * k * c

                self.wo = self.wo + dw

            # Store the output of the system.
            zt[ti] = self.z
            wo_len[ti] = np.sqrt(self.wo.dot(self.wo))

        error_avg = np.sum(np.abs(zt-ft))/simtime_len
        print('Training MAE: {:.3f}'.format(error_avg))

        return zt, wo_len


    def predict(self, start, end, dt):

        simtime = np.arange(start, end, dt)
        simtime_len = len(simtime)

        zpt = np.zeros(simtime_len)

        x = self.x
        r = np.activation(x)
        z = self.wo.dot(r)

        for ti, t in enumerate(simtime):

            # sim, so x(t) and r(t) are created.
            x = (1.0-dt)*x + self.M.dot(r*dt) + self.wf*(z*dt)
            r = self.activation(x)
            z = self.wo.dot(r)

            zpt[ti] = z

        return t, zpt

    def evaluate(self, start, end, dt, func_learned):

        t, zpt = self.predict(start, end, dt)

        simtime_len = len(zpt)

        ft = func_to_learn(t)

        error_avg = np.sum(np.abs(zpt-ft))/simtime_len
        print('Testing MAE: {:.3f}'.format(error_avg))

        return error_avg
