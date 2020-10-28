import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

#Synthesize some data
mu, sigma = 5, 1
points1 = np.random.normal(mu, sigma, size = (100, 2) )

mu, sigma = 2, 1
points2 = np.random.normal(mu, sigma, size = (100, 2) )


x = np.random.normal(1, 1, size = 100)
y = np.random.normal(8, 1, size = 100)
points3 = np.column_stack((x, y))


x = np.row_stack((points1, points2, points3))

#Implement k-means
n_iter = 30
def norm(x):
    return sum([i*i for i in x])

#Create random cluster centroids
n_clusters = 3
mu = x[np.random.randint(x.shape[0], size = n_clusters)]

#Initiate assignment matrix

n_points = x.shape[0]
for iteration in  tqdm(range(n_iter)):

    # E-step (assign points to centroids)
    distances = np.array([np.sum(np.square(np.array(x - i)), axis = 1) for i in mu])
    min_idx = np.argmin(distances, axis =0)
    r = np.zeros((n_points, n_clusters))
    r[np.arange(min_idx.size),min_idx] = 1
    cluster = np.argmax(r, axis = 1)


    # M-step (update centroids to minimise cost)
    mu = (np.dot(x.T, r)/sum(r)).T

#Plot Results
cluster = np.argmax(r, axis = 1)
plt.figure()
plt.scatter(x[:,0], x[:, 1], c=cluster, cmap='plasma')
plt.scatter(mu[:, 0], mu[:, 1], c = "y", edgecolors='b',s = 100)
plt.show()


