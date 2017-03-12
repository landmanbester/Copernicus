import numpy as np

class MCMC_diagnostics(object):
    """
    A class to compute the Gelman-Rubin diagnostic
    """
    def __init__(self, Nsamplers, sampslist):
        """

        :param Nsamplers:  The number of MCMC chains
        :param sampslist: A list containing samples of each chain
        """
        self.M = Nsamplers
        try:
            self.D, self.T = sampslist[0].shape
        except:
            self.D = 1
            self.T = sampslist[0].size
        self.n = self.T #Can be used to discard a fraction of the samples
        self.samps = np.zeros([self.M, self.n, self.D])  # array to store samps (one page for each chain with each column holding samples of a parameter)
        for i in xrange(self.M):
            self.samps[i] = sampslist[i][:,0:self.n].T #np.asarray(holder[holder.files[0]])  # [self.n:self.T,:])

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