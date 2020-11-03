import numpy as np
from scipy.stats import dirichlet
from scipy.special import gamma


class DirichletMixture:
    """
    Description
    -----------

    The parameters alpha_i for a finite dirichlet mixture are estimated using EM. 
    The variable names correspond to the following quantities:
    pi - mixing probability for a given component
    Z - the latent variable of dimensions N x D that denotes the assignment vector
    """

    def __init__(self, k, max_iter=5):
        self.k = k
        self.max_iter = int(max_iter)

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full( shape=self.shape, fill_value=1/self.k)
        
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [  X[row_index,:] for row_index in random_row ]
        self.sigma = [ np.cov(X.T) for _ in range(self.k) ]

    def e_step(self, X):
        # E-Step: calculate delta for the current values of alpha and pi
        self.delta = self.calc_delta(X)

    
    def m_step(self, X):
        # M-Step: update mu and sigma holding phi and weights constant
        for i in range(self.k):
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
        delta = np.zeros( (self.n, self.k) )
        # Simplify the implementation by naming smaller terms and vectorise as much as possible
        
        delta = numerator / denominator
        return delta
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)
