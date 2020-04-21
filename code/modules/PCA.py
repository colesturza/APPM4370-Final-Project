from modules.FORCE import Force
import numpy as np
from scipy.sparse import random
from sklearn.decomposition import PCA

class PCA_NN(Force):

################################################################################
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
        dW_out = np.zeros((self.N, 1))

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
        zt = np.zeros((simtime_len, 1))

    #Okay so now we are leanrning

        x = 0.5*np.random.randn(self.N, 1)
        z = 0.5*np.random.randn()

        r = self.activation(x)

        P = (1.0/alpha)*np.eye(self.N) #Inverse correlation matrix

        pca = PCA(n_components=100)

        projections = []

        X = []

        #Iterate and train the network
        for ti in range(simtime_len):

            # sim, so x(t) and r(t) are created.
            x = (1.0-dt)*x + self.W_int.dot(r*dt) + self.W_feed.dot(z)*dt
            r = self.activation(x)
            z = self.W_out.T.dot(r)

            X.append(x.reshape(self.N).tolist())

            # # Perform PCA at each timestep and project w onto PC 1, PC 2, and PC 80.
            # pca.fit(np.array(X).dot(np.array(X).T)/(self.N * self.N))
            #
            # eigvects = pca.components_
            #
            # # Project w onto PC 1
            # project_pc1 = self.W_out.T.dot(eigvects[0])
            #
            # # Project w onto PC 2
            # project_pc2 = self.W_out.T.dot(eigvects[1])
            #
            # # Project w onto PC 80
            # project_pc80 = self.W_out.T.dot(eigvects[79])
            #
            # projections.append([project_pc1, project_pc2, project_pc80])

            if (ti+1) % learn_every == 0:
                #Update inverse correlation matrix
                k = P.dot(r)
                rPr = r.T.dot(k)
                c = 1.0/(1.0 + rPr)
                P = P - np.outer(k, k * c)

                #Update the error for the linear readout
                e = z - ft[ti]

                #Update the output weights
                dW_out = -k.dot(e.T) * c
                self.W_out = self.W_out + dW_out

            #Store the output of the system.
            zt[ti,:] = z

        #Average error after learning
        error_avg = np.sum(np.abs(np.subtract(zt, ft)))/simtime_len
        print('Training MAE: {:.5f}'.format(error_avg))

        X = np.array(X)

        # subtract the mean of each column
        X = X - np.mean(X, axis=0)

        # Sample variance-covariance matrix
        S = X.T.dot(X)/self.N

        print(S.shape)

        pca.fit(S)
        eigvals = pca.explained_variance_ # calculate variance ratios
        eigvects = pca.components_

        #Return the training progression
        return zt, x, eigvals, eigvects, projections
