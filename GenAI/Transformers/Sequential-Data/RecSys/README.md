
# Transformers For Recommendation Sys

Transformers4Rec is an open source repo, [**NVIDIA-Merlin/Transformers4Rec**](https://github.com/NVIDIA-Merlin/Transformers4Rec), from Nvidia. 

It is a flexible and efficient library for sequential and **session-based recommendation** and can work with PyTorch.

## Sequential recommendation

In recommendation, input data is typically a sequence of interactions such as items that are browsed in a web session or items put in a cart. A recommendation system helps you process and model the interactions so that you can output better recommendations for the next item.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/Sequential-Data/RecSys/images/sequential_rec.png" width="800">


Traditional recommendation algorithms usually ignore the temporal dynamics and the sequence of interactions when trying to model user behavior. Generally, the next user interaction is related to the sequence of the user's previous choices. In some cases, it might be a repeated purchase or song play. User interests can also suffer from interest drift because preferences can change over time. Those challenges are addressed by the **sequential recommendation** task.


## Session-based recommendation with Transformers4Rec

A special use case of **sequential-recommendation** is the **session-based recommendation** task where you only have access to the short sequence of interactions within the current session. This is very common in online services like e-commerce, news, and media portals where the user might choose to browse anonymously due to GDPR compliance that restricts collecting cookies or because the user is new to the site. This task is also relevant for scenarios where the users' interests change a lot over time depending on the user context or intent. In this case, leveraging the interactions for the current session is more promising than old interactions to provide relevant recommendations.

To deal with sequential and session-based recommendation, many sequence learning algorithms previously applied in machine learning and NLP research have been explored for RecSys based on k-Nearest Neighbors, Frequent Pattern Mining, Hidden Markov Models, Recurrent Neural Networks, and more recently neural architectures using the Self-Attention Mechanism and transformer architectures. Unlike Transformers4Rec, these frameworks only accept sequences of item IDs as input and do not provide a modularized, scalable implementation for production usage.

## Technology

Transformers4Rec has a first-class integration with Hugging Face (HF) Transformers, NVTabular, and Triton Inference Server, making it easy to build end-to-end GPU accelerated pipelines for sequential and session-based recommendation.

Refer the info page: [End-to-End Pipeline with Hugging Face Transformers and NVIDIA Merlin](https://nvidia-merlin.github.io/Transformers4Rec/stable/pipeline.html).

### Integration with Hugging Face Transformers

Transformers4Rec integrates with Hugging Face Transformers, allowing RecSys researchers and practitioners to easily experiment with the latest state-of-the-art NLP Transformer architectures for sequential and session-based recommendation tasks and deploy those models into production.

Models are composed of three building blocks:
* Tokenizer that converts raw text to sparse index encodings
* Transformer architecture
* Head for NLP tasks such as text classification, generation, sentiment analysis, translation, and summarization

### Pipeline

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/Sequential-Data/RecSys/images/pipeline.png" width="800">