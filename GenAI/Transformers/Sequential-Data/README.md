# Transformers to Sequential Behavior Data

Applying Transformer architectures to **tabular**, **sequential data** — rather than just text or images—is exactly how industry leaders like Stripe and Amazon are currently pushing the boundaries of fraud detection and personalization.

Encoding a sequence of **user behaviors** (like swipe, transfer, click, purchase) requires **treating transactions like words in a sentence**, where the "grammar" is the user's baseline behavior, and a sudden deviation in that grammar might indicate fraud or signal a new purchase intent. 

When you start coding, you will find that the hardest part isn't the Transformer itself—it is the **Tokenization**. In NLP, tokenizing words is standard. However, in behavioral data, you have to decide how to represent continuous variables (like a transaction amount of $42.50) and categorical variables (like Merchant_ID) into a single discrete token or a unified embedding space before it ever touches the attention mechanism.

Below we list relevant open-source repositories and public datasets tailored specifically for the use cases.

## Transaction Foundation Model For Fraud Detection

[Transaction-Foundation-Model](https://github.com/HsiangHung/Machine-Learning/tree/master/GenAI/Transformers/Sequential-Data/Transaction-Foundation-Model)

If you want to build something analogous to Stripe’s Payments Foundation Model, this is the exact blueprint you need.

* **Repository**: NVIDIA-AI-Blueprints/transaction-foundation-model
* **What it shows**: This repo provides a complete, step-by-step Jupyter Notebook workflow for building a transaction foundation model. It covers:
    * How to convert raw tabular transaction records into domain-specific token sequences
    * Pretrain a decoder-only foundation model using causal language modeling
    * Extract the behavioral embeddings
    * Feed those embeddings into an XGBoost model to classify fraud
* **Public Data**: The blueprint natively uses the TabFormer dataset (originally created by IBM), which is a massive, publicly available synthetic dataset containing 24 million credit card transactions.

## Transformers4Rec For Recommendations

[Transformers4Rec](https://github.com/HsiangHung/Machine-Learning/tree/master/GenAI/Transformers/Sequential-Data/RecSys)


If you want to encode behavioral sequences to predict a user's next action (e.g., product recommendations), you should look into the Merlin ecosystem.

* **Repository**: NVIDIA-Merlin/Transformers4Rec
* **What it shows**: This is a flexible library built on PyTorch and Hugging Face Transformers specifically designed for sequential and session-based recommendations. It automatically handles the heavy lifting of merging context features with sequential features, allowing you to use over 64 different Transformer architectures (like BERT, GPT-2, or XLNet) to predict a user's next interaction based on their historical timeline.
* **Public Data**: Their examples folder provides end-to-end tutorials using public e-commerce datasets like Yoochoose and REES46.

## Time-Series Transformer

[Time-Series-Forecast](https://github.com/HsiangHung/Machine-Learning/tree/master/GenAI/Transformers/Sequential-Data/Time-Series-Forecast)

Transformers can be also used for time-series forecasting problem. By [GeeksforGeeks: Transformer for time series forecasting](https://www.geeksforgeeks.org/deep-learning/transformer-for-time-series-forecasting/), there are some Pros and Cons:

* Advantages:
    * **Models long range dependencies**: Transformers use self attention mechanism to directly connect any two points in the input sequence no matter how far apart they are as they capture trends and seasonal patterns over **long horizons**.
    * **Scalability**: With innovations like Informer, Performer, or Reformer transformers can scale to very long sequences at manageable computational cost.
    * **Unified architecture**: Transformers can incorporate different modalities like **categorical features**, **input embeddings** and tasks like forecasting, in real world problems.
    * **Handles missing or irregular data**
* Disadvantages
    * **Quadratic complexity $O(n^2)$**: Data points in a series must be multiplied by **every other data point** in the series as each data point we add to input increases the time it takes to calculate attention. It is quadratic complexity.
    * **High Costs**: Due to quadratic time and memory complexity with respect to the sequence length due to full self-attention. For a long time series data this can be computationally expensive and memory-intensive.
    * **Need large data**: Transformers need large amounts of training data to capture the patterns and make predictions. Small or noisy time series datasets can cause overfitting or poor model performance.
    * **Complex model design**: The architecture of time series transformers can be complex as it involves multiple components like input embeddings, positional encodings, and sometimes hybrid layers. This complexity leads to longer experimentation cycles and harder hyperparameter tuning.


