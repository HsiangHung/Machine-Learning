"""
Here is a complete example of how to build and query an embedding model using PyTorch. 
In a real-world scenario, this embedding layer would be the first layer of a larger downstream model 
(like a Deep Neural Network or an Autoencoder for UEBA).

Preparing the Input (The Vocabulary Map):
Crucial Engineering Step: You must reserve an index (usually 0) for an <UNK> (Unknown) token.

Choosing the Embedding Dimension:
A standard rule of thumb (popularized by Jeremy Howard and FastAI) for determining the embedding 
dimension (D) based on the cardinality (C) of the variable is D = min(50, C/2)

Alternatively, for massively high cardinality (millions of IDs), you often use the 
fourth-root rule: D = C^{0.25}

So, for 1,000,000 unique emails, the embedding dimension would be approximately 32.
"""
import torch
import torch.nn as nn


def get_vocab(emails):
    # Build the vocabulary mapping (String -> Integer)
    # Index 0 is reserved for <UNK> (Unknown / Out of Vocabulary)
    vocab = {"<UNK>": 0}
    for email in emails:
        if email not in vocab:
            vocab[email] = len(vocab)
    
    return vocab


# 2. Define the Model Architecture
class EntityEmbeddingModel(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(EntityEmbeddingModel, self).__init__()
        # nn.Embedding is essentially a lookup table of trainable weights
        self.email_embedding = nn.Embedding(
            num_embeddings=num_users, 
            embedding_dim=embedding_dim
        )
        
        # In a real model, you would have downstream layers here
        # self.fc1 = nn.Linear(embedding_dim, 64)
        
    def forward(self, email_indices):
        # The forward pass converts integer indices into dense vectors
        return self.email_embedding(email_indices)


if __name__ == "__main__":

    # 1. Simulate the Preprocessing Stage
    raw_emails = ["alice@company.com", "bob@company.com", "charlie@company.com", 
                  "bob@company.com", "david_new_employee@company.com"]

    vocab = get_vocab(raw_emails)

    cardinality = len(vocab) # 4 (3 emails + 1 UNK)
    embedding_dim = 16 # Chosen based on the formula above

    # Initialize the model
    model = EntityEmbeddingModel(num_users=cardinality, embedding_dim=embedding_dim)

    # 3. Inference / Forward Pass Example
    # We have a batch of two emails: one known, one brand new.
    incoming_batch = ["alice@company.com", "bob@company.com", "charlie@company.com", 
                    "jsmith@company.com", "adoe@company.com"] # unknown email

    # Map incoming strings to integers, defaulting to 0 (<UNK>) if not found
    input_indices = [vocab.get(email, 0) for email in incoming_batch]
    input_tensor = torch.tensor(input_indices, dtype=torch.long)

    print(f"Input Indices: {input_tensor}")

    # Pass through the model
    dense_vectors = model(input_tensor)

    print(f"Output Embedding Shape: {dense_vectors.shape}")
    print(dense_vectors)
    # we can last two unknown emails have same embeddings
