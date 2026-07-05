
# Transformers For Recommendation Sys

Transformers4Rec is an open source repo, [**NVIDIA-Merlin/Transformers4Rec**](https://github.com/NVIDIA-Merlin/Transformers4Rec), from Nvidia. 

It is a flexible and efficient library for sequential and **session-based recommendation** and can work with PyTorch.

Input data is typically a sequence of interactions such as items that are browsed in a web session or items put in a cart. The library helps you process and model the interactions so that you can output better recommendations for the next item.


<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/Sequential-Data/RecSys/images/sequential_rec.png" width="900">

## Sequential and Session-based recommendation with Transformers4Rec

Traditional recommendation algorithms usually ignore the temporal dynamics and the sequence of interactions when trying to model user behavior. Generally, the next user interaction is related to the sequence of the user's previous choices. In some cases, it might be a repeated purchase or song play. User interests can also suffer from interest drift because preferences can change over time. Those challenges are addressed by the **sequential recommendation** task.

A special use case of **sequential-recommendation** is the session-based recommendation task where you only have access to the short sequence of interactions within the current session. This is very common in online services like e-commerce, news, and media portals where the user might choose to browse anonymously due to GDPR compliance that restricts collecting cookies or because the user is new to the site. This task is also relevant for scenarios where the users' interests change a lot over time depending on the user context or intent. In this case, leveraging the interactions for the current session is more promising than old interactions to provide relevant recommendations.

To deal with sequential and session-based recommendation, many sequence learning algorithms previously applied in machine learning and NLP research have been explored for RecSys based on k-Nearest Neighbors, Frequent Pattern Mining, Hidden Markov Models, Recurrent Neural Networks, and more recently neural architectures using the Self-Attention Mechanism and transformer architectures. Unlike Transformers4Rec, these frameworks only accept sequences of item IDs as input and do not provide a modularized, scalable implementation for production usage.