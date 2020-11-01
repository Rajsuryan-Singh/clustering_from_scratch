## <ins>K-means<ins>


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

## <ins>Gaussian Mixture Model<ins>
Data is modelled as a mixture of gaussians. Once we have defined how many gaussians are used and what their parameters and relative probabilities are, the simple generative model would do the following two things:

    1. Pick a gaussian based on the relative probability (essentially their weight in a multinomial distribution)
    2. Generate a point based on the probability distribution of that gaussian

If repeated enough times, this will generate a distribution that is the weighted sum of all the gaussians involved. 

The gaussians can be multivariate and can have different means and covariance matrices. 

We follow a similar approach here as well. We initate random gaussians. We introduce a latent variable that is a 1-of-k type n-dimensional vector (n is the number of gaussians). This variable, say z, represents which gaussian was a point drawn from. We also introduce a pseudo parameter that is the probability of having drawn a point from the k-th gaussian. 

Then we do the iteration until convergence optimising for the log likelihood function where at the first step we keep the gaussian parameters constant and adjust the weights. In the second steps we keep the weights constant and adjust the parameters of the gaussian. 

This is also assured to converge. This is a class of algorithms called coordinate ascent. 

**Need to revisit this with a better hold on linear algebra and probability theory.**

## <ins>Dirichlet Mixture<ins>
### Why do we need it? 
* Gaussians can't model assymetric distributions
* If the data comes from the multinomial space, for example an amino acid in a protein at a given position or a note in a song at a given time, dirichlet distributions take into account the non negative constraints and are a conjugate prior to the multinomial distribution. So dirichlet distribution makes the idea canditate for modelling such feature spaces. 

### Clustering using the dirichlet mixture model

Not many resources available on the internet. People have done it using EM [1], Gibbs sampling [2] and SEM [3]. 

The approach implemented here is the SEM, the hybrid SEM as the authors have named it. It works on the generalised dirichlet distribution where we allow our random variables to be positively correlated [4]. In the dirichlet distribution, all the random variables are negatively correlated. This might not be the case in observed data and it has been shown to be otherwise [4] in the case of biological data. Intuitively speaking, the negative correlation only holds if the data is of a  multinomial nature i.e. chosing the occurence of one automatically ensures that the others don't occur. There can be data that would be conviniently modelled by the dirichlet distribution but do not adhere to this multinomial quality. Notes in music (which can be represented by a 12 element vector at any given time) are a good example of such data. Notes are bound to be positively correlated as there are multiple notes in any given chord and any occurence of that chord means the occurence of all of those notes at the same time. The use of the generalised dirichlet distribution would be appropriate in such cases.




## References 
1. Sjölander, K., Karplus, K., Brown, M., Hughey, R., Krogh, A., Mian, I. S., & Haussler, D. (1996). Dirichlet mixtures: a method for improved detection of weak but significant protein sequence homology. Computer applications in the biosciences : CABIOS, 12(4), 327–345. https://doi.org/10.1093/bioinformatics/12.4.327

1.  Ye, Xugang et al. “On the inference of dirichlet mixture priors for protein sequence comparison.” Journal of computational biology : a journal of computational molecular cell biology vol. 18,8 (2011): 941-54. doi:10.1089/cmb.2011.0040

3. N. Bouguila and D. Ziou, "A hybrid SEM algorithm for high-dimensional unsupervised learning using a finite generalized Dirichlet mixture," in IEEE Transactions on Image Processing, vol. 15, no. 9, pp. 2657-2668, Sept. 2006, doi: 10.1109/TIP.2006.877379.

4. R. J. Connor and J. E. Mosimann, “Concepts of independence for proportions with a generalization of the Dirichlet distribution,” Amer. Statist. Assoc. J., vol. 64, pp. 194–206, 1969.