import numpy as np
from scipy.linalg import block_diag

class NormalDistribution:
    """
    Representation of a multi-variate normal distribution
    """
    def __init__(self, mu, sigma, **meta):
        """
        Construct a normal distribution
        :param mu: The mean
        :param sigma: The covariance matrix
        """
        self.mu = mu
        self.sigma = sigma
        self.meta = meta
        self.isigma = np.linalg.pinv(sigma)
        self.logd = np.log(np.linalg.det(sigma))

    @staticmethod
    def from_mfcc(mfcc_mean, mfcc_covar, chroma_stft_covar,chroma_stft_mean,H_mean,H_var,P_mean,P_var,**meta):
        """
        Construct normal distribution from mfcc parameters
        :param mfcc_mean: ffcc mean
        :param mfcc_covar: mfcc covariance matrix
        :param id: id of distribution
        :param meta: Meta data
        :return: The constructed normal distribution
        """
        def lower_to_full(low, N):
            C = np.zeros((N,N))
            C[np.tril_indices(N)] = low
            C += C.T
            np.fill_diagonal(C, C.diagonal() / 2)
            return C

        n1=5
        n2=5
        mu=np.append(np.append(np.array(mfcc_mean)[:n1],np.array(chroma_stft_mean[:n2])),[H_mean,P_mean])
        sigma=block_diag(lower_to_full(mfcc_covar[:int((n1*(n1+1))/2)], n1),
                         lower_to_full(chroma_stft_covar[:int((n2*(n2+1))/2)], n2),
                         H_var,P_var)
        return NormalDistribution(mu, sigma,**meta)

    @property
    def n(self):
        """
        Dimension property
        :return: The dimension
        """
        return len(self.mu)

    def skl(self, other):
        """
        Computes the symmetrized Kullback-Leibler divergence between
        two normal ditributions.

        :param other: Another normal distribution
        :return: The symmetrized Kullback-Leibler divergence
        """
        dmu = self.mu-other.mu
        isigma = self.isigma + other.isigma
        return ((self.sigma*other.isigma).sum() +
                (self.isigma*other.sigma).sum() +
                dmu.dot(isigma).dot(dmu)
                - 2*self.n)

    def log_skl(self, other):
        return np.log(1+self.skl(other))

    def sqrt_skl(self, other):
        return np.sqrt(self.skl(other))

    def js(self, other):
        """
        Computes the a Jensen-Shannon-like divergence where
        the mean of the two normal distributions are approximated
        by another normal distribution.

        :param other: Another normal distribution
        :return: A Jensen-Shannon-like divergence
        """
        sigma_m = 0.5*(self.sigma + other.sigma + np.outer(self.mu, self.mu) + np.outer(other.mu, other.mu)) \
            - 0.25*np.outer(self.mu + other.mu, self.mu + other.mu)
        return 2*np.log(np.linalg.det(sigma_m)) - self.logd - other.logd

    def sqrt_js(self, other):
        return np.sqrt(self.js(other))

    def __dict__(self):
        return {'mu': self.mu,
                'sigma':  self.sigma}




