from modules.FORCE import Force
import numpy as np
from scipy.sparse import random
from sklearn import preprocessing

def PCA(data, n_components=8, return_matrix=True, return_eigenvalues=True):

    # Covariance matrix
    cov_matrix = np.cov(preprocessing.scale(data.T))

    # Diagonalization of the covariance matrix
    eig_val, eig_vec = np.linalg.eigh(cov_matrix)

    if return_matrix or return_eigenvalues:

        if return_matrix:
            # Projection of the data points over the eigenvectors
            Proj = data.dot(eig_vec[:,-n_components:])

        if return_matrix and return_eigenvalues:
            return eig_vec[:,-n_components:], Proj, eig_val

        elif return_matrix:
            return eig_vec[:,-n_components:], Proj

        else:
            return eig_vec[:,-n_components:], eig_val

    return eig_vec[:,-n_components:]

class PCA_Network(Force):

################################################################################
    def fit(self, simtime, func_to_learn, *, alpha=1.0, n_components=8):

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

        ft = ft.reshape((simtime_len, 1))

        #Essentially the output function (vector)
        zt = np.zeros((simtime_len, self.readouts))

    #Okay so now we are leanrning

        #x is pre-activation and z is readout
        #x has to be initialized somewhere
        #NOTE: Why is z random as well
        if self.x is None:
            x = self.setIC()
        else: x = self.x

        #post-activation
        r = self.activation(x)

        if self.rand_z:
            z = 0.5*np.random.randn(self.readouts, 1)
        else: z = self.W_out.T.dot(r)

        P = (1.0/alpha)*np.eye(self.N) #Inverse correlation matrix

        #Magnitude of weights as we learn
        proj_ws = np.zeros((simtime_len, n_components, self.readouts))

        # Store x after activation (r) at each time step
        X = np.zeros((simtime_len, self.N))

        #Iterate and train the network
        for ti in range(simtime_len):

            x = x - (x*dt + self.W_int.dot(r*dt) + self.W_feed.dot(z)*dt)/self.tau
            r = self.activation(x)
            z = self.W_out.T.dot(r)

            #Store r at time step ti
            X[ti, :] = r.reshape(self.N)

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
            #
            # if ti%100 == 0 and ti != 0:
            #     print('Finished {}'.format(ti))

        self.x = x #Keep the final state

        eigvecs, proj, eigvals = PCA(X, n_components=n_components, return_matrix=True, return_eigenvalues=True)

        # Projection over the leading principal components
        proj_w = self.W_out.T.dot(eigvecs)

        z_proj = proj.dot(proj_w.T)

        eigvals = eigvals[::-1]

        proj = proj.T[::-1,...]

        #Return the training progression
        return z_proj, eigvals, proj, proj_ws
