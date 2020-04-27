from modules.FORCE import Force
import numpy as np
from scipy.sparse import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

        # PCA stuff
        pca = PCA(n_components=100)
        projections = np.zeros((simtime_len, 9, self.readouts))
        X = np.zeros((simtime_len, self.N))

        #Iterate and train the network
        for ti in range(simtime_len):

            # sim, so x(t) and r(t) are created.
            x = (1.0-dt)*x + self.W_int.dot(r*dt) + self.W_feed.dot(z)*dt
            r = self.activation(x)
            z = self.W_out.T.dot(r)

            X[ti, :] = x.reshape(self.N)

            # Perform PCA at each timestep and project w onto PC 1, PC 2, and PC 80.
            X_t = X[:ti+1,:]
            # Sample variance-covariance matrix
            S = X_t.T.dot(X_t)
            #S_std = sc.fit_transform(S)
            pca.fit(S)#S_std)

            eigvects = pca.components_

            # Project w onto PC 1
            project_pc1 = self.W_out.T.dot(eigvects[0]).reshape(self.readouts)
            # Project w onto PC 2
            project_pc2 = self.W_out.T.dot(eigvects[1]).reshape(self.readouts)
            # Project w onto PC 3
            project_pc3 = self.W_out.T.dot(eigvects[2]).reshape(self.readouts)
            # Project w onto PC 4
            project_pc4 = self.W_out.T.dot(eigvects[3]).reshape(self.readouts)
            # Project w onto PC 5
            project_pc5 = self.W_out.T.dot(eigvects[4]).reshape(self.readouts)
            # Project w onto PC 6
            project_pc6 = self.W_out.T.dot(eigvects[5]).reshape(self.readouts)
            # Project w onto PC 7
            project_pc7 = self.W_out.T.dot(eigvects[6]).reshape(self.readouts)
            # Project w onto PC 8
            project_pc8 = self.W_out.T.dot(eigvects[7]).reshape(self.readouts)
            # Project w onto PC 80
            project_pc80 = self.W_out.T.dot(eigvects[79]).reshape(self.readouts)

            projects = np.array([project_pc1, project_pc2, project_pc3,
                                 project_pc4, project_pc5, project_pc6,
                                 project_pc7, project_pc8, project_pc80])

            projections[ti, :] = projects

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

            #Saving internal outputs
            if self.saveInternal:
                self.intOut[ti,:] = r[:self.num2save,0]

            #Store the output of the system.
            zt[ti,:] = z.reshape(self.readouts)

            if ti % 100 == 0:
                print('Finished {}'.format(ti))

        #Average error after learning
        self.x = x #Keep the final state
        error_avg = np.sum(np.abs(np.subtract(zt, ft)))/simtime_len
        print('Training MAE: {:.5f}'.format(error_avg))

        # Sample variance-covariance matrix
        S = X_t.T.dot(X_t)
        #S_std = sc.fit_transform(S)
        pca.fit(S)#S_std)
        eigvals = pca.explained_variance_

        #Return the training progression
        return zt, x, eigvals, projections

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

        # PCA stuff
        pca = PCA(n_components=100)
        sc = StandardScaler()
        projections = np.zeros((simtime_len, 9))
        X = np.zeros((simtime_len, self.N))

        for ti in range(simtime_len):

            # sim, so x(t) and r(t) are created.
            x = (1.0-dt)*x + self.W_int.dot(r*dt) + self.W_feed.dot(z)*dt
            r = self.activation(x)
            z = self.W_out.T.dot(r)

            # Perform PCA
            X[ti, :] = x.reshape(self.N)

            # Perform PCA at each timestep and project w onto PC 1, PC 2, and PC 80.
            X_t = X[:ti+1,:]
            # Sample variance-covariance matrix
            S = X_t.T.dot(X_t)
            S_std = sc.fit_transform(S)
            pca.fit(S_std)

            eigvects = pca.components_

            # Project w onto PC 1
            project_pc1 = self.W_out.T.dot(eigvects[0])[0]
            # Project w onto PC 2
            project_pc2 = self.W_out.T.dot(eigvects[1])[0]
            # Project w onto PC 3
            project_pc3 = self.W_out.T.dot(eigvects[2])[0]
            # Project w onto PC 4
            project_pc4 = self.W_out.T.dot(eigvects[3])[0]
            # Project w onto PC 5
            project_pc5 = self.W_out.T.dot(eigvects[4])[0]
            # Project w onto PC 6
            project_pc6 = self.W_out.T.dot(eigvects[5])[0]
            # Project w onto PC 7
            project_pc7 = self.W_out.T.dot(eigvects[6])[0]
            # Project w onto PC 8
            project_pc8 = self.W_out.T.dot(eigvects[7])[0]
            # Project w onto PC 80
            project_pc80 = self.W_out.T.dot(eigvects[79])[0]

            projects = np.array([project_pc1, project_pc2, project_pc3,
                                 project_pc4, project_pc5, project_pc6,
                                 project_pc7, project_pc8, project_pc80])

            projections[ti, :] = projects

            zpt[ti,:] = z.reshape(self.readouts)

        # Sample variance-covariance matrix
        S = X.T.dot(X)
        S_std = sc.fit_transform(S)
        pca.fit(S_std)
        eigvals = pca.explained_variance_

        return zpt, eigvals, projections

################################################################################
    #Evaluate the neural network
    #NOTE: Need to consider multiple readouts and inputs
    #NOTE: Should check on all of this stuff
    def evaluate(self, x, simtime, func_learned):

        zpt = super().predict(x, simtime)

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
