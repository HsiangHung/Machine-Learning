
# Active Learning


Active Learning is a special case of Supervised Machine Learning, which allows a learning algorithm to interactively query a user to label data with the desired outputs. The algorithm actively chooses from the pool of unlabeled data the subset of examples to be labelled next in active learning. 

The basic idea is to deliberately select the situations where the model is **unsure** of itself or where it could most benefit from more information.

Where should we apply active learning?
* We have a very small amount or a huge amount of dataset.
* Annotation of the unlabeled dataset costs human effort, time, and money.
* We have access to limited processing power.



Reference:
* [Geeksforgeeks - ML | Active Learning](https://www.geeksforgeeks.org/machine-learning/ml-active-learning/)


## Process

Here's a detailed explanation of how active learning usually works:

1. **Initialization**: To train an initial machine learning model, start with a small labelled dataset.
2. **Model Training**: Using the available labelled data, train the first model.
3. **Uncertainty Estimation**: 
    * To predict the unlabeled data, apply the trained model.
    * Calculate the model's prediction confidence or uncertainty. Margin, variance, and entropy are examples of common metrics.
4. **Query Technique**:
    * Choose the cases from the unlabeled pool where the model is unsure or has low confidence.
    * Depending on the particular active learning algorithm, the query strategy selected may include picking the cases with the **highest level of uncertainty**, cases close to the decision boundary, or cases where models in an ensemble disagree.
5. **Labelling**: Ask an oracle-a human annotator or another source of ground truth labels—for **labels for the chosen instances**.
6. **Model Update**:
    * Add the recently annotated data to the training set.
    * Using the revised labeled dataset, retrain the model.
7. **Repeat**: Repeat steps 2 through 6 iteratively until a budget is depleted or a performance threshold is reached.


## Key concepts of Active Learning

* **Query Strategy**: It is important to have a strategy in place for deciding which instances to query for labels. Diverse approaches concentrate on different elements, like diversity, uncertainty, or representative sampling.
* **Model Uncertainty**: Active learning makes use of the uncertainty estimates that models frequently provide to pinpoint situations in which the model is unsure of its predictions.
* **Oracle**: The organisation in charge of the ground truth labelling. In actuality, this might be an outside system or a human annotator.
* **Stopping Conditions**: A model's performance may plateau, a target accuracy may be met, or the iteration of active learning may end when a predetermined number of labelled examples is reached.

