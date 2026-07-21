
# High Cardinality Feaure Embedding

One-hot encoding for high cardinality features, e.g. 
* email
* id
* zipcode

is not practical, which will create unnecessary messive columns. 

We need to resort to embeddings the features.


* `encode_high_cardinatlity.py` is an example to encode high cardinality features. However, this script needs to load all values in vocab to generate embeddings. This may lead to issue in memeory.

* `hash_bucket_encode.py` uses hash Embeddings when you have massive cardinality, no sequential relationship, and need to avoid memory blowouts (e.g., embedding raw IP addresses or device MACs)

* `sequental_encode.py` use sequential embeddings (Item2Vec) when you have a "bounded" vocabulary 
(like 5,000 specific SaaS event types) and want the neural network to mathematically understand that certain actions belong to similar workflows.

