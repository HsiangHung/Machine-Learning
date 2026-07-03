# Transaction Foundation Model

## Under This Directory

We build a domain-specific foundation model that learns **sequential transaction patterns** via **self-supervised pre-training**, and show that combining its learned embeddings with raw features significantly improves fraud detection precision.

By pretraining on large volumes of unlabeled transaction sequences, we can learn general-purpose representations of financial behavior that transfer to a wide range of downstream tasks:
* Fraud detection
* Anomaly scoring
* Customer segmentation
* Personalized financial services

and so on.

## Technology

In this example, we follow the code from Nvidia repo [NVIDIA Developer Example: Build Your Own Transaction Foundation Model](https://github.com/HsiangHung/transaction-foundation-model/tree/a2fb683917f47f6e44582dad994925e96155f836). In this repo:
* **Custom GPU-accelerated tokenizer** — A modular, RAPIDS-powered tokenizer converts heterogeneous tabular fields (merchant category, amount, time deltas, and more) into domain-specific token sequences.
* **Scalable pretraining with NeMo AutoModel** — A decoder-only foundation model is trained with causal language modeling through [NVIDIA NeMo AutoModel](https://github.com/nvidia-nemo/automodel). 
* Embedding extraction and downstream evaluation — Learned embeddings are extracted via last-token pooling and evaluated on fraud detection with XGBoost, 


## Notebooks

These notebooks are all run using Google Colab. For third notebook to run NeMo, we even use A100 high ram option to tentatively train the model with 100 steps.


| # | Notebook |	Description |
| --- | --- | --- |
| 1 | N1-XGBoost-Fraud-Detection.ipynb | Load the TabFormer financial transaction dataset, create temporal train/val/test splits, and train a GPU-accelerated XGBoost baseline for fraud detection. |
| 2 | N2-Sequential-Preprocessing-Tokenization.ipynb | Build a custom GPU-accelerated tokenizer pipeline that converts transaction records into domain-specific token sequences. | 
| 3 | N-3Foundation=Model-Training.ipynb | Pretrain a decoder-only foundation model (~29M parameters) on tokenized transaction sequences using NeMo AutoModel with causal language modeling. |
