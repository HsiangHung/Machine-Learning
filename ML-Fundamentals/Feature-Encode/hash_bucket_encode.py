"""
Hash Embeddings

The Problem: 
Traditional embeddings (nn.Embedding) require you to store a dictionary mapping every 
unique string (e.g., user_id, email) to an integer index. If you have 100 million unique users, 
just holding that dictionary and the massive embedding matrix in RAM will crash your system. 
Furthermore, it completely fails on "Cold Start" (new users unseen during training) because you 
have no index for them.

The Solution:
Hash Embeddings remove the dictionary entirely. Instead of looking up an index, you run the raw 
string through a deterministic hashing algorithm (like MurmurHash or MD5). You take the hash 
value modulo $K$ (where $K$ is the maximum number of embedding rows you want to store in memory) 
to get the row index.

* Pro: Zero memory spent on a vocabulary dictionary. Instantly handles unseen, out-of-vocabulary data.
* Con: Hash Collisions. Two different users might hash to the exact same row index and share the same embedding vector.

* Why Collisions are Okay: Neural networks are surprisingly resilient to this. 
If hacker@evil.com and CEO@company.com collide, the neural network learns to use the other features 
in the dataset (like IP address, action type) to distinguish the risk, treating the shared embedding 
as just one noisy feature among many.

Use Hash Embeddings when you have massive cardinality, no sequential relationship, and need to avoid 
memory blowouts (e.g., embedding raw IP addresses or device MACs)
"""

import torch
import torch.nn as nn
import hashlib

class HashEmbedding(nn.Module):
    def __init__(self, num_buckets, embedding_dim):
        """
        num_buckets: The maximum size of the embedding table (K).
                     Even if you have 1B users, you can set this to 10M.
        embedding_dim: The size of the dense vector.
        """
        super(HashEmbedding, self).__init__()
        self.num_buckets = num_buckets
        
        # We still use nn.Embedding under the hood, but we control the inputs
        self.embedding = nn.Embedding(
            num_embeddings=num_buckets, 
            embedding_dim=embedding_dim
        )
        
    def _hash_string_to_int(self, string_val):
        """
        Converts a string into an integer between 0 and (num_buckets - 1).
        Using MD5 for consistency across platforms.
        """
        # Convert string to bytes, hash it, take the first 8 bytes (64 bit int)
        hash_bytes = hashlib.md5(string_val.encode('utf-8')).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='little')
        
        # Modulo K to fit within our bucket size
        return hash_int % self.num_buckets

    def forward(self, string_batch):
        """
        Takes a list of raw strings, hashes them on the fly, and returns embeddings.
        """
        # 1. Hash the raw strings on the fly (No vocabulary dictionary needed!)
        hashed_indices = [self._hash_string_to_int(val) for val in string_batch]
        
        # 2. Convert to tensor
        input_tensor = torch.tensor(hashed_indices, dtype=torch.long)
        
        # 3. Lookup embeddings
        return self.embedding(input_tensor)


if __name__ == "__main__":

    # --- Example Usage ---
    # We have millions of users, but we force them into 10,000 buckets
    hash_layer = HashEmbedding(num_buckets=10000, embedding_dim=16)

    # Notice we pass RAW STRINGS directly to the model
    batch = ["alice@company.com", "bob@company.com", "unknown_user_55@company.com",
             "jsmith@company.com", "adoe@company.com"] 
    # all are referred as unknown emails
    embeddings = hash_layer(batch)

    print("Hashed Embeddings Shape:", embeddings.shape) 
    # Output: torch.Size([3, 16])
    print(embeddings)