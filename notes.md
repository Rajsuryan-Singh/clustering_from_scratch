## K-means


Simple algorithm to separate unlabelled data into clusters. 

The algorithm:

We have a bunch of points. We select n random cluster centroids and assign and create an assignment matrix where we assign the points closest to a cluster centroid to that cluster. The closeness is determined by a creterion that we define in that space, most common is the euclidian distance. 

We also define a cost function as the sum of sqared distances between points and the centroids. 

Then we go through an EM like algorithm where we repeat the following steps until comvergence:

    1. Keep the cluster centres constant and update the cluster membership of points to minimise the cost function. 
    2. Keep the membership matrix constant and update the cluster centroids in order to minimise the cost function. 
    It can be proved that the cost function will be minimised if we update the cluster centroid to be the mean of all the points in the given cluster by taking the partial derivative of the cost function wrt the cluster centroids and equating it to zero. 

This algorithm is bound to converge because at the value of the cost function is decreasing at every step. But there's a probability that this can get stuck in a local optima subject to the starting conditions. This can be taken care of by running it a few times with different starting conditions and use the results with the minimum cost. For well defined clusters and properly chosen cluster number, the algorith getting stuck at local optima wouldn't be very likely. 

<div style="page-break-after: always; visibility: hidden"> 
\pagebreak 
</div>

## Gaussian Mixture Model
Data is modelled as a mixture of gaussians. Once we have defined how many gaussians are used and what their parameters and relative probabilities are, the simple generative model would do the following two things:

    1. Pick a gaussian based on the relative probability (essentially their weight in a multinomial distribution)
    2. Generate a point based on the probability distribution of that gaussian

If repeated enough times, this will generate a distribution that is the weighted sum of all the gaussians involved. 

The gaussians can be multivariate and can have different means and covariance matrices. 

We follow a similar approach here as well. We initate random gaussians. We introduce a latent variable that is a 1-of-k type n-dimensional vector (n is the number of gaussians). This variable, say z, represents which gaussian was a point drawn from. We also introduce a pseudo parameter that is the probability of having drawn a point from the k-th gaussian. 

Then we do the iteration until convergence optimising for the log likelihood function where at the first step we keep the gaussian parameters constant and adjust the weights. In the second steps we keep the weights constant and adjust the parameters of the gaussian. 

This is also assured to converge. This is a class of algorithms called coordinate ascent. 

**Need to revisit this with a better hold on linear algebra and probability theory.**