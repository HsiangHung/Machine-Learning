# UMAP

UMAP (aka Uniform Manifold Approximation and Projection) is a powerful dimension reduction technique that has gained significant traction in the fields of machine learning and data visualization, by Leland McInnes, John Healy, and James Melville.

UMAP is often compared to t-SNE (t-distributed Stochastic Neighbor Embedding) due to its similar application in data visualization, but it offers several advantages, including better preservation of global data structure and faster computation times.

### Reference

* [GeeksforFeeks, UMAP: Uniform Manifold Approximation and Projection](https://www.geeksforgeeks.org/machine-learning/umap-uniform-manifold-approximation-and-projection/)


## Mathematical Foundations


UMAP is built on solid mathematical foundations, including Riemannian geometry and algebraic topology, grounded in several key mathematical concepts:

* **Riemannian Manifold**: UMAP assumes that the data is uniformly distributed on a Riemannian manifold. This means that the data points lie on a smooth, curved surface that can be locally approximated by Euclidean space.
* **Riemannian Metric**: The Riemannian metric is locally constant or can be approximated as such. This metric defines the distance between points on the manifold.
* **Topological Data Analysis**: UMAP leverages topological data analysis to capture the structure of the data. It constructs a fuzzy topological representation of the data, which is then optimized to find a low-dimensional embedding.


## The Algorithm

The UMAP algorithm can be broken down into two main phases: constructing a fuzzy topological representation and optimizing the low-dimensional embedding.

### 1. Constructing the Fuzzy Topological Representation

Nearest Neighbor Search: UMAP begins by finding the nearest neighbors for each data point. This is typically done using approximate nearest neighbor algorithms to speed up the process.
Fuzzy Simplicial Set: A fuzzy simplicial set is constructed from the nearest neighbors. This set captures the local connectivity of the data points.
Fuzzy Membership Strengths: Membership strengths are assigned to the edges of the simplicial set, representing the probability that two points are connected.

### 2. Optimizing the Low-Dimensional Embedding

Cross-Entropy Optimization: The low-dimensional embedding is optimized to minimize the cross-entropy between the fuzzy simplicial set in the high-dimensional space and the low-dimensional space.
Stochastic Gradient Descent: UMAP uses stochastic gradient descent to perform the optimization. This involves iteratively adjusting the positions of the points in the low-dimensional space to better match the fuzzy topological structure.

