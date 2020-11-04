import numpy as np
from scipy.stats import dirichlet
from scipy.special import gamma
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

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

    def __init__(self, n_clusters, max_iter=200, threshold = 0.001):
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.max_iter = int(max_iter)

    def init_alpha(self, X):
        # Crude estimate for the values of alpha based on the initialised clusters
        # Adapted from the matlab script for fastfit by Thomas P. Minka

        E = np.mean(X, axis = 0)
        E2 = np.mean(np.square(X), axis = 0)
        return ((E[0] - E2[0]) / (E2[0] - np.square(E[0]))) * E

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape
        
        self.pi = np.zeros(self.n_clusters)
        self.delta = np.zeros((self.n, self.n_clusters))
        self.alpha = np.zeros((self.n_clusters, self.m ))

        #Modify using k-means 
        kmeans = KMeans(n_clusters = self.n_clusters).fit(X)
        labels = kmeans.labels_
        X_subsets = dict()

        #Divide the data into subsets to estimate the alphas for each cluster
        for i in range(self.n_clusters):
            X_subsets[i] = X[labels == i]
        
        #Estimate alpha for each cluster
        for i in range(self.n_clusters):
            self.alpha[i, :] = self.init_alpha(X_subsets[i])

        #Estimate pi using the initial cluster assignments
        for i in range(self.n_clusters):
            self.pi[i] = np.sum(labels == i)/len(labels)
        


    def e_step(self, X):
        # E-Step: calculate delta for the current values of alpha and pi
        self.delta = self.calc_delta(X)

    
    def m_step(self):
        # M-Step: update alpha and pi for the current values of delta
        self.pi_new = np.sum(self.delta, axis = 0)/(self.n)
        temp = (self.X_proj/(self.sum_x_alpha -1))
        temp = np.transpose(temp, axes = [2, 0, 1])
        update_numerator = np.sum(np.multiply(self.delta , temp), axis = 1)
        update_denominator = np.sum(np.multiply(self.delta, self.sum_data/(self.sum_all - 1)), axis = 0)
        update_factor = update_numerator/update_denominator
        update_factor = update_factor.T
        self.alpha = self.alpha * update_factor
        


    def fit(self, X):
        self.initialize(X)
        
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step()

            #Check if converged
            if np.sum(np.square(self.pi - self.pi_new)) < self.threshold:
                return

            self.pi = self.pi_new
            
            
    def calc_delta(self, X):
        X_proj = np.broadcast_to(X,(self.n_clusters,)+X.shape) 
        X_proj = np.swapaxes(X_proj, 0, 1)

        gamma_alpha = gamma(self.alpha)

        alpha_temp = np.broadcast_to(self.alpha, (self.n,)+ self.alpha.shape)

        sum_x_alpha = np.add(X_proj, alpha_temp)
        gamma_sum = gamma(sum_x_alpha)

        term1 = np.prod(gamma_sum, axis = 2)

        sum_alpha = np.sum(self.alpha, axis = 1)
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

        #update variables to be reused in the m-step
        self.X_proj = X_proj
        self.sum_x_alpha = sum_x_alpha
        self.sum_data = sum_data
        self.sum_all = sum_all

        return delta
    
    def predict(self, X):
        delta = self.calc_delta(X)
        return np.argmax(self.delta, axis=1)

#Test on synthetic data
alpha1 = 10, 1
points1 = np.random.dirichlet(alpha1, size = (100) )

alpha2 = 1,10
points2 = np.random.dirichlet(alpha2, size = (100) )


alpha3 = 30,30
points3 = np.random.dirichlet(alpha3, size = (100) )


x = np.row_stack((points1, points2, points3))

#Fit Model

dmm = DirichletMixture(n_clusters=3, max_iter=100, threshold = 0.001)
dmm.fit(x)
cluster = dmm.predict(x)
plt.figure()
plt.scatter(x[:,0], x[:, 1], c=cluster, cmap='plasma')
plt.show()
