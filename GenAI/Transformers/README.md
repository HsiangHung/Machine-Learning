# Transformer

It all start from the seminal paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The transformer architecture from the paper:

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/images/seminal_transformer_arch.png" width="600">


The transformer architecture can replace traditional models like LSTMs and GRUs with self attention mechanisms in language models due to crucial advantages: 
* The ability to capture **long range** dependencies in text
* **Parallel** processing that enables faster training
* Unprecedented **scalability** that allows building increasingly powerful models.

Transformers are first used for language model, used for ChatGPT and translation.
Here we list good resources to explain transformer:

* Quick tutorial: [ransformers Step-by-Step Explained (Attention Is All You Need)](https://www.youtube.com/watch?v=avjX3QrYkls)
* [Decoder-Only Transformers, ChatGPTs specific Transformer, Clearly Explained!!!](https://www.youtube.com/watch?v=bQ5BoolX9Ag)
* [Vizuara lab: The Transformers](https://www.vizuaranewsletter.com/p/the-transformers?r=5b5pyd&utm_campaign=post&utm_medium=web)


## Multi-head attention

The following nice discussion is brought by [Vizuara lab: The Transformers](https://www.vizuaranewsletter.com/p/the-transformers?r=5b5pyd&utm_campaign=post&utm_medium=web) to explain why we need multi-head attention in attention layers.

### Limitations of Self-Attention Mechanisms

A significant problem with a single self-attention mechanism is its limited ability to effectively handle linguistic ambiguity. This challenge can be illustrated with the sentence:
```
The artist painted the portrait of a woman with a brush.
```

This statement has two distinct and valid interpretations:
* The first interpretation is that the artist used a brush as a tool to perform the action of painting. In this context, the phrase “with a brush” modifies the verb “painted”.
* The second interpretation is that the subject of the painting is a woman who is holding a brush. Here, “with a brush” modifies the “woman” or “portrait”.

A single self-attention layer may struggle to capture both of these potential relationships simultaneously. It might incorrectly average these dependencies or fixate on only one, resulting in a contextual vector that fails to represent the full nuance of the sentence.

### Intuition of Multi-Head Attention

The solution is to use multiple self-attention mechanisms in parallel, aka multi-head attention. The same input embedding matrix is fed into several independent self-attention “heads”. Each head produces its own distinct context vector matrix, effectively learning a different set of relationships or focusing on a different aspect of the input, such as one head capturing verb-centric relationships while another captures a different semantic nuance.

By google Genmi, as the model trains, these heads naturally specialize like:
* Head 1 might learn to strictly look for Subject-Verb relationships.
* Head 2 might learn to strictly look for adjectives and the nouns they modify.
* Head 3 might become an expert at figuring out what pronouns refer to.
* Head 4 might look at the immediate previous word to keep track of sequence order.
* ....


## Variants of Transformers

### Translator

See [Encoder + Decoder Translator](https://github.com/HsiangHung/Machine-Learning/tree/master/GenAI/Transformers/LLM/Translators).

### Vision Transformers

Transformers are later used for image models. For example, we can patch an image to sequences of tokens.


<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/images/vision_transformer_arch.png" width="800">

Here we build few models to demo vision-transformer implementation in ML:
1. Image classification model
2. Text-Image Clip model

### Video Vision Transformer

* [GeeksforGeeks: Video Vision Transformer (ViViT)](https://www.geeksforgeeks.org/computer-vision/video-vision-transformer-vivit/)
* [Medium: Video Transformer(VIT): A Deep Learning Model for Video Processing](https://medium.com/@nadav6stern/video-transformer-vit-a-deep-learning-model-for-video-processing-442268c8c3b4)


### Diffusion Transformer


<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/images/diffusion_transformer_arch.png" width="900">


### Behavior Transformers

#### Treat Transactions Like Language

[Stripe’s Payments Foundation Model: When Transactions Learned to Speak](https://medium.com/@mumbaiyachori/stripes-payments-foundation-model-when-transactions-learned-to-speak-30c87111cb31)


#### Treat User & Customer Behaviour Like Language

Idea: language could be applied to the analysis of user behavior. In language problem, we “play” with predicting the next word in a sentence to create wonders like text generation or tools that describe language, such as embeddings. Here author tried to predict the next page view, the next purchase, or the next user action using a sufficiently large dataset. [Beha2Vec — Using Transformers to Analyze User & Customer Behaviour](https://pdellov.medium.com/beha2vec-using-transformers-to-analyze-user-customer-behaviour-34d9f45b652a)


The author used the old-fashioned Google Merchandise Store dataset, a demo dataset available through Kaggle. It is basically the dataset created over the Google Merchandise Store, which Google itself uses to demo its Google Analytics product. It contains 2+ GiB of user navigation events tracked via GA4. These include anonymous user_pseudo_id (cookie IDs) tied to actions like page views and purchases.

The ustom transformer model dramatically increases the performance.

Code: https://github.com/pdellov/beha2vec


### Heterogeneous Graph Transformers

See teh medium post: [Building a Fraud Detection Model using Graph Neural Networks (GNNs)](https://natashagluons.medium.com/building-a-fraud-detection-model-using-graph-neural-networks-gnns-d3c62b7c38e9) and the demo code on github: [Syndicate Indication using Network Graph Analytics](https://github.com/natgluons/Syndicate-Indication-using-Network-Graph-Analytics).
