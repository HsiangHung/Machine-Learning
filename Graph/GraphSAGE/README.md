# GraphSAGE (SAmple and aggreGatE)

All the following content follows the blog [How GraphSAGE Handles Changing Graph Structure](https://towardsdatascience.com/graph-neural-networks-part-3-how-graphsage-handles-changing-graph-structure/).


For super large graphs it’s computationally impossible to process all neighbors of a node (except if you have limitless time, which we all don’t…), like with traditional GCNs. 

GraphSAGE makes training much faster and scalable by:
* Sampling only a **subset** of neighbors.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/Graph/GraphSAGE/images/graphSAGE_sampling.png" width="700">

* Combining the features of the sampled neighbors with an aggregation function. 


## Sampling Neighbors

Sampling is easy for tabular data, when creating train, test, and validation sets. With graphs, you cannot select random nodes. This can result in disconnected graphs, nodes without neighbors. For example, like:

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/Graph/GraphSAGE/images/random_select_node.png" width="600">

## Aggregate Information

After the neighbor selection from the previous part, GraphSAGE combines their features into one single representation. The most common aggregation types and the ones explained in the paper are:
* Mean aggregation
* LSTM
* Pooling 


### Mean Aggregation

The average is computed over all sampled neighbors’ features (very simple and often effective), defined as

$$h_N=\frac{1}{|N(v)|}\sum_{u \in N(v)} h_u,$$

where $h_u$ is feature of neighbor $u$ from the sampled neighbor set $N(v)$.

### LSTM Aggregation 

LSTM aggregation uses an LSTM (type of neural network) to process neighbor features sequentially. It can capture more complex relationships, and is more powerful than mean aggregation. 

### Pool Aggregation

applies a non-linear function to extract key features (think about max-pooling in a neural network, where you also take the maximum value of some values).


## Node Representation

After sampling and aggregation, the node combines its previous features with the aggregated neighbor features. Nodes will learn from their neighbors but also keep their own identity, just like we saw before with Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs). 

$$h^{\prime}_v = \sigma \Big( W \dot CONCAT(h_v, AGGREGATE(h_u, u \in N(v))) \Big),$$

where $W$ is the weight matrix, $CONCAT$ is concatenate vectors and $AGGREGATE$ is the aggregated features from previous section.

The aggregation of step 2 is done over all neighbors, and then the feature representation of the node is concatenated. This vector is multiplied by the weight matrix, and passed through non-linearity (for example ReLU). As a final step, normalization can be applied.


## Repeat for Multiple Layers

The first three steps can be repeated multiple times, when this happens, information can flow from distant neighbors. In the image below you see a node with three neighbors selected in the first layer (direct neighbors), and two neighbors selected in the second layer (neighbors of neighbors). 

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/Graph/GraphSAGE/images/repeat_multilayer.png" width="700">


In short, GraphSAGE has it scalability and flexibility. Aggregation helps with generalization because it smooths out noisy features. The multi-layers allow the model to learn from far-away nodes.




