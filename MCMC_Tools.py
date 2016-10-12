import numpy as np

class MCMC_diagnostics(object):
    """
    A class to compute the Gelman-Rubin diagnostic
    """
    def __init__(self, Nsamplers, sampslist):
        """

        :param Nsamplers:  The number of MCMC chains
        :param funct_to_test: The function whose GR diagnostic to compute
        :param fname: the directory in which the results are saved
        """
        self.M = Nsamplers
        # Load the first samples (to get Nsamps etc)
        #holder = np.load(fname + "Samps0" + '.npz')[func_to_test]
        #print holder.shape
        self.D, self.T = sampslist[0].shape
        self.n = self.T/2 #Discards the first half of the parameters
        self.samps = np.zeros([self.M, self.n, self.D])  # array to store samps (one page for each chain with each column holding samples of a parameter)
        #self.samps[0] = holder[:, 0:self.n].T
        #self.X = np.zeros([self.M * self.n, self.D])
        for i in xrange(self.M):
            #holder = np.load(fname + "Samps" + str(i) + '.npz')[func_to_test]
            self.samps[i] = sampslist[i][:,0:self.n].T #np.asarray(holder[holder.files[0]])  # [self.n:self.T,:])
            #self.X[i * self.n:(i + 1) * self.n, :] = self.samps[i]

    def get_GRC(self):
        W = np.zeros(self.D)
        B = np.zeros(self.D)
        # Get the means of each parameter in each chain
        meansj = np.zeros([self.M, self.D])
        for i in range(self.M):
            meansj[i, :] = np.mean(self.samps[i], axis=0)
            W += np.sum((self.samps[i] - meansj[i, :]) ** 2, axis=0)
        W = W / (self.n - 1) / self.M
        # Now get the overall means
        means = np.mean(meansj, axis=0)
        # print meansj
        # print means
        # Compute B
        B = self.n * np.sum((meansj - means) ** 2, axis=0) / (self.M - 1)
        self.meansj = meansj
        self.means = means
        return np.sqrt(((self.n - 1) * W / self.n + B / self.n) / W)