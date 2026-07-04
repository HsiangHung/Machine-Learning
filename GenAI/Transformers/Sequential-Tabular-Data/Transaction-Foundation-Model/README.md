# Transaction Foundation Model

## Under This Directory

We build a domain-specific foundation model that learns **sequential transaction patterns** via **self-supervised pre-training**, and show that combining its learned embeddings with raw features significantly improves fraud detection precision.

By pretraining on large volumes of unlabeled transaction sequences, we can learn general-purpose representations of financial behavior that transfer to a wide range of downstream tasks:
* Fraud detection
* Anomaly scoring
* Customer segmentation
* Personalized financial services

and so on.

## Finanical Tokenization

While traditional machine learning models evaluate a single transaction row in isolation, a Transaction Foundation Model strings hundreds of transactions together in chronological order to form a longitudinal timeline of a user's life.

Here is how the tokenization breaks down into two distinct levels to capture that timeline.

### 1. The Micro-Sequence (Within a Record)

First, the model must convert a single tabular row into something a Transformer can read. It takes the disparate fields of a single transaction (like the merchant, the transaction amount, and the time of day) and flattens them into a short "phrase" of discrete tokens.

For example, a single $15 coffee purchase might become a micro-sequence of tokens:
["Merchant_Category_Cafe", "Amount_Bucket_10_20", "Day_Tuesday", "Time_Morning"]

The data schema we use here has 13 features per transaction (e.g., Merchant ID, Amount, Time, Card Type, etc.), and the serialization script maps each feature to exactly 1 token, then every single transaction record consumes exactly 13 tokens in the sequence.


### 2. The Macro-Sequence (Across Records)

Once those individual transactions are flattened into micro-sequences, they are concatenated together in chronological order to represent the user's historical behavior. Then the funnel analogy of financial events applies

The sequence fed into the Transformer looks like a continuous story:
[...Transaction 1 Tokens...] -> [...Transaction 2 Tokens...] -> [...Transaction 3 Tokens...] 

Thus, a complete temporal transaction looks like 

$$ <bos> AMT_1 MERCH_1 ... CUST_1 <sep> AMT_2 MERCH_2 ... CUST_2 <sep>  AMT_3 MERCH_3 ... CUST_3 ... <eos> $$

The transaction foundation model allows a single 4,096-token sequence. It allows packs in roughly 315 ($\sim$ 4096/13) consecutive transactions for a single account.


## Technology

In this example, we follow the code from the Nvidia repo: [NVIDIA Developer Example: Build Your Own Transaction Foundation Model](https://github.com/HsiangHung/transaction-foundation-model/tree/a2fb683917f47f6e44582dad994925e96155f836). In this repo:
* **Custom GPU-accelerated tokenizer** — A modular, RAPIDS-powered tokenizer converts heterogeneous tabular fields (merchant category, amount, time deltas, and more) into domain-specific token sequences.
* **Scalable pretraining with NeMo AutoModel** — A decoder-only foundation model is trained with causal language modeling through [NVIDIA NeMo AutoModel](https://github.com/nvidia-nemo/automodel). 
* Embedding extraction and downstream evaluation — Learned embeddings are extracted via last-token pooling and evaluated on fraud detection with XGBoost, 


## Notebooks

These notebooks are all run using Google Colab. For third notebook to run NeMo, we even use A100 high ram option to tentatively train the model with 100 steps.


| # | Notebook |	Description |
| --- | --- | --- |
| 1 | **n1_XGBoost_fraud_detection.ipynb** | Load the TabFormer financial transaction dataset, create temporal train/val/test splits, and train a GPU-accelerated XGBoost baseline for fraud detection. |
| 2 | **n2_sequential_preprocessing_tokenization.ipynb** | Build a custom GPU-accelerated tokenizer pipeline that converts transaction records into domain-specific token sequences. | 
| 3 | **n3_foundation_model_training.ipynb** | Pretrain a decoder-only foundation model (~29M parameters) on tokenized transaction sequences using NeMo AutoModel with causal language modeling. |
| 4 | **n4_inference_embedding_extraction.ipynb** | Load the pretrained model, run GPU inference, extract 512-dimensional embeddings via last-token pooling, and visualize with UMAP. |
