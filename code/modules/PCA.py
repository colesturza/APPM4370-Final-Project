from modules.FORCE import Force
import numpy as np
from scipy.sparse import random
from sklearn import preprocessing

# Perform PCA on a given set of data
# n_components = the number of leading principal components to keep
# return_proj = return projection of data points over the eigenvectors if True
# return_eigenvals = return leading principal components' eigenvalue if True
def PCA(data, *, n_components=8, return_proj=True, return_eigenvals=True):

    # Create covariance matrix after scaling data
    cov_matrix = np.cov(preprocessing.scale(data.T))

    # Diagonalization of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    if return_proj or return_eigenvals:

        if return_proj:

            # Projection of the data points over the eigenvectors
            proj = data.dot(eigvecs[:,-n_components:])

            if return_eigenvals:

                return eigvecs[:,-n_components:], proj, eigvals

            else:

                return eigvecs[:,-n_components:], proj

        else:

            return eigvecs[:,-n_components:], eigvals

    return eigvecs[:,-n_components:]

class PCA_Network(Force):

################################################################################
#Use the trained neural network predict or generate
#NOTE: Need to consider multiple readouts and inputs
    def predict(self, simtime, n_components=8):

        if self.x is None:
            x = self.setIC()
        else: x = self.x

        simtime_len = simtime.shape[0]

        diff = np.diff(simtime)
        if not np.any(np.isclose(diff, diff[0])):
            raise ValueError('All values in simtime must be evenly spaced.')

        dt = diff[0]

        if self.saveInternal:
            self.intOut = np.zeros((simtime_len, self.num2save))

        zpt = np.zeros((simtime_len, self.readouts))

        r = self.activation(x)
        z = self.W_out.T.dot(r)

        # Store x after activation (r) at each time step
        X = np.zeros((simtime_len, self.N))

        for ti in range(simtime_len):

            # sim, so x(t) and r(t) are created.
            # x = ((1.0-dt)*x + self.W_int.dot(r*dt) + self.W_feed.dot(z)*dt)/self.tau
            x = x - (x*dt + self.W_int.dot(r*dt) + self.W_feed.dot(z)*dt)/self.tau
            r = self.activation(x)
            z = self.W_out.T.dot(r)

            if self.saveInternal:
                self.intOut[ti,:] = r[:self.num2save,0]

            #Store r at time step ti
            X[ti, :] = r.reshape(self.N)

        eigvecs, proj, eigvals = PCA(X, n_components=n_components, return_proj=True, return_eigenvals=True)

        # Projection over the leading principal components
        proj_w = self.W_out.T.dot(eigvecs) # proj_w readouts x pc components kept

        # z_proj simtime_len x readouts
        z_proj = proj.dot(proj_w.T)

        eigvals = eigvals[::-1]

        proj = proj.T[::-1,...]

        return z_proj, eigvals, proj

################################################################################
    #Evaluate the neural network
    #NOTE: Need to consider multiple readouts and inputs
    #NOTE: Should check on all of this stuff
    def evaluate(self, x, simtime, func_learned):

        z_proj, _, _ = self.predict(x, simtime)

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

        error_avg = np.sum(np.abs(np.subtract(z_proj, ft)))/simtime_len
        print('Testing MAE: {:.5f}'.format(error_avg))

        return error_avg
