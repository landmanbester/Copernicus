import numpy as np

class MCMC_diagnostics(object):
    """
    A class to compute the Gelman-Rubin diagnostic
    """
    def __init__(self, M, T, D, samps):
        self.M = M  # The number of chains
        self.T = T  # First interval at which they are saved (each subsequent interval doubles)
        self.D = D  # Number of parameters
        self.n = self.T  # /2
        self.samps = np.zeros([M, self.n, self.D])  # array to store samps (one page for each chain with each column holding samples of a parameter)
        self.X = np.zeros([self.M * self.n, self.D])
        tmpstr = ['0', '1', '2', '3', '4']
        i = 0
        for tmp in tmpstr:
            holder = np.load(savename + tmp + '.npz')
            self.samps[i] = asarray(holder[holder.files[0]])  # [self.n:self.T,:])
            self.X[i * self.n:(i + 1) * self.n, :] = self.samps[i]
            i += 1

    def get_GRC(self):
        W = zeros(self.D)
        B = zeros(self.D)
        # Get the means of each parameter in each chain
        meansj = zeros([self.M, self.D])
        for i in range(self.M):
            meansj[i, :] = mean(self.samps[i], axis=0)
            W += sum((self.samps[i] - meansj[i, :]) ** 2, axis=0)
        W = W / (self.n - 1) / self.M
        # Now get the overall means
        means = mean(meansj, axis=0)
        # print meansj
        # print means
        # Compute B
        B = self.n * sum((meansj - means) ** 2, axis=0) / (self.M - 1)
        self.meansj = meansj
        self.means = means
        return sqrt(((self.n - 1) * W / self.n + B / self.n) / W)