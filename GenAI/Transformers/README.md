# Transformer

It all start from the seminal paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The transformer architecture from the paper:

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/images/seminal_transformer_arch.png" width="500">

Transformers are first used for language model, used for ChatGPT and translation.
Here we list good resources to explain transformer:

* Quick tutorial: [ransformers Step-by-Step Explained (Attention Is All You Need)](https://www.youtube.com/watch?v=avjX3QrYkls)
* [Decoder-Only Transformers, ChatGPTs specific Transformer, Clearly Explained!!!](https://www.youtube.com/watch?v=bQ5BoolX9Ag)
* [Vizuara lab: The Transformers](https://www.vizuaranewsletter.com/p/the-transformers?r=5b5pyd&utm_campaign=post&utm_medium=web)

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


### Heterogeneous Graph Transformers

See teh medium post: [Building a Fraud Detection Model using Graph Neural Networks (GNNs)](https://natashagluons.medium.com/building-a-fraud-detection-model-using-graph-neural-networks-gnns-d3c62b7c38e9) and the demo code on github: [Syndicate Indication using Network Graph Analytics](https://github.com/natgluons/Syndicate-Indication-using-Network-Graph-Analytics).
