
# Anomaly Detection Using AutoEncoder

We can train an autoencoder model only on normal dataset, and define the reconstruct errors as anomaly scores.

The idea is that, we build the normal score distribution, and the hypothesis is that fraud may have different score distribution.

The study here follows the medium post [Autoencoders and Testing their Potential in Anomaly Detection](https://medium.com/@amnahhmohammed/autoencoders-and-testing-their-potential-in-anomaly-detection-09135140fd56).

As we can see below, the normal datasets show very different distribution behavior as normal ones.

## Normal 

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/Anomaly-Detection/autoencoder/images/normal_anomaly_score.png" width="700">


## Normal vs Abnormal

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/Anomaly-Detection/autoencoder/images/fault_vs_normal_anomaly_score.png" width="700">