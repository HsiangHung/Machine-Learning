"""
Sequential Embeddings (Item2Vec)

The Problem: 
Standard embeddings treat categorical IDs as independent. But in UEBA or Recommendation Systems, 
actions happen in a sequence. If an employee accesses Repo_A, then Repo_B, then Repo_C, those 
three repositories are semantically related in the context of that user's workflow.

The Solution: 
We borrow the Skip-Gram architecture from NLP (Word2Vec) to create Item2Vec. We treat the user's 
daily sequence of actions as a "sentence" and the actions/items as "words."

We train a model to predict the surrounding context actions given a target action. 
This forces actions that frequently appear in the same user sessions to cluster together 
in the embedding space.

Use Sequential Embeddings (Item2Vec) when you have a "bounded" vocabulary 
(like 5,000 specific SaaS event types) and want the neural network to mathematically understand that 
certain actions belong to similar workflows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Item2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Item2Vec, self).__init__()
        
        # The Target Embedding (e.g., the action the user just took)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # The Context Embedding (the actions surrounding the target)
        # Note: Often, weights are tied (target_embeddings = context_embeddings) 
        # but in strict Word2Vec, they are separate matrices.
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, target_item, context_item):
        """
        target_item: Tensor of shape (Batch_size,)
        context_item: Tensor of shape (Batch_size,)
        Returns a score predicting how likely context_item is to appear near target_item.
        """
        # Shape: (Batch_size, Embedding_Dim)
        target_vec = self.target_embeddings(target_item)
        context_vec = self.context_embeddings(context_item)
        
        # Dot product between target and context vectors
        # Shape: (Batch_size,)
        score = torch.sum(target_vec * context_vec, dim=1)
        
        return score


if __name__ == "__main__":
    # --- Example Training Loop Concept ---
    vocab_size = 5000 # Number of distinct SaaS actions
    embedding_dim = 32

    model = Item2Vec(vocab_size, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Assume we processed logs and generated pairs: (Target_Action, Context_Action)
    # e.g., (github_clone, aws_s3_read)
    # Positive labels (1) mean they actually appeared together in a session.
    target_batch = torch.tensor([14, 52, 99])   
    context_batch = torch.tensor([52, 14, 102]) 
    labels = torch.tensor([1.0, 1.0, 1.0]) 

    # In a real scenario, you MUST also generate Negative Samples (random pairs with label 0)
    # to prevent the model from just pushing all embeddings to the same point.

    # Training Step
    optimizer.zero_grad()
    predictions = model(target_batch, context_batch)

    # We use BCEWithLogitsLoss because we want to push the dot product 
    # of positive pairs to +infinity and negative pairs to -infinity
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(predictions, labels)

    loss.backward()
    optimizer.step()

    # --- Post-Training Usage ---
    # After training, you throw away the context_embeddings.
    # You extract the target_embeddings matrix.
    # Now, github_clone (Index 14) and aws_s3_read (Index 52) will be close in vector space!
    final_item_vectors = model.target_embeddings.weight.data
