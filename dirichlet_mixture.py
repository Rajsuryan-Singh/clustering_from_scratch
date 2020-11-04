import numpy as np
from scipy.stats import dirichlet
from scipy.special import gamma


class DirichletMixture:
    """
    Description
    -----------
    X should have dimensions - N x dim where N is the number of points each point 
    being a dim dimensional vector

    The parameters alpha_i for a finite dirichlet mixture are estimated using EM. 
    The variable names correspond to the following quantities:
    pi - mixing probability for a given component
    Z - the latent variable of dimensions N x D that denotes the assignment vector
    """

    def __init__(self, n_clusters, max_iter=200):
        self.n_clusters = n_clusters
        self.max_iter = int(max_iter)

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape
        
        self.pi = np.zeros(self.n_clusters)
        self.delta = np.zeros((self.n, self.n_clusters))
        self.alpha = np.zeros((self.n_clusters, self.m ))

        #Modify using k-means


    def e_step(self, X):
        # E-Step: calculate delta for the current values of alpha and pi
        self.delta = self.calc_delta(X)

    
    def m_step(self, X):
        # M-Step: update alpha and pi for the current values of delta
        for i in range(self.n_clusters):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, 
                aweights=(weight/total_weight).flatten(), 
                bias=True)

    def fit(self, X):
        self.initialize(X)
        
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            
    def calc_delta(self, X):
        X_temp = np.broadcast_to(X,(self.n_clusters,)+X.shape) 
        X_temp = np.swapaxes(X_temp, 0, 1)

        gamma_alpha = gamma(self.alpha)

        alpha_temp = np.broadcast_to(alpha, (self.n,)+ alpha.shape)

        sum_x_alpha = np.add(X_temp, alpha_temp)
        gamma_sum = gamma(sum_x_alpha)

        term1 = np.prod(gamma_sum, axis = 2)

        sum_alpha = np.sum(alpha, axis = 1)
        sum_data = np.sum(X , axis = 1)

        sum_alpha = np.broadcast_to(sum_alpha, (self.n,) + sum_alpha.shape)
        sum_data = np.broadcast_to(sum_data, (self.n_clusters,) + sum_data.shape)
        sum_data = np.swapaxes(sum_data, 0, 1)

        sum_all = sum_alpha + sum_data

        gamma_sum_alpha = gamma(sum_alpha)
        gamma_sum_all = gamma(sum_all)

        term2 = np.divide(gamma_sum_alpha, gamma_sum_all)

        numerator = np.multiply(term1, term2)
        numerator = np.multiply(numerator, self.pi)
        denominator = np.sum(numerator, axis = 1).reshape(-1, 1)
        delta = numerator / denominator
        return delta
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)
