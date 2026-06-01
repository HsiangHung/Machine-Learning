
# Classification Model Using Vision-Transformer

In this example, we are going to build a **classification model** with **vision transformers** from scratch, and test is on the **handwritten digits MNIST dataset**.

The codes here follows the Medium blog: [Building a Vision Transformer Model From Scratch](https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6).

The vision transformer model uses the standard approach of adding a **learnable classification token** to the patch embeddings in order to perform classification.

Transformers take embeddings in parallel. While this increases the speed, transformers are not aware of what order sequences are supposed to be in. In order to fix this problem, positional encodings need to be added to the patch embeddings.