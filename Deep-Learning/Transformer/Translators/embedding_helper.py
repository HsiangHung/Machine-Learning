import math
import torch
import torch.nn as nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    
    def __init__(self, emb_size, max_len=5000, dropout=None):
        """
        emb_size = The dimension of the transformer, which is also the number of embedding values per token.
                  In the transformer I used in the StatQuest: Transformer Neural Networks Clearly Explained!!!
                  emb_size=2, so that's what we'll use as a default for now.
                  However, in "Attention Is All You Need" emb_size=512
        max_len = maximum number of tokens we allow as input.
                  Since we are precomputing the position encoding values and storing them in a lookup table
                  we can use emb_size and max_len to determine the number of rows and columns in that
                  lookup table.
        
                  In this simple example, we are only using short phrases, so we are using
                  max_len=6 as the default setting.
                  However, in The Annotated Transformer, they set the default value for max_len to 5000

        positional encoding:
         * PE(pos, 2i)   = sin(pos/10000^{2i/d})
         * PE(pos, 2i+1) = cos(pos/10000^{2i/d})

         pos: token position (e.g. "what is StatQuest", pos["what"]=0)
         i: stands for embedding. (e.g. "what" word embedding = [0.2, -0.1, 1,2, -0.4], i = [0, 0, 2, 2]
        """
        super().__init__()

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1) # i, [0, 1, 2, 3....]
        embedding_index = torch.arange(start=0, end=emb_size, step=2).float() # 2*i, [0, 2, 4, 6, ...]
       
        div_term = 1/torch.tensor(10000.0)**(embedding_index / emb_size) # 1/10000^{2i/d} 
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # "pe" will look like Tensor([[0, 1], [0.84, 0.53], ...])

        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        self.register_buffer('pe', pe) ## "register_buffer()" ensures that 'pe' will be moved to wherever the model gets
                                       ## moved to. So if the model is moved to a GPU, then, even though we don't need to optimize 'pe', 
                                       ## it will also be moved to that GPU. This, in turn, means that accessing 'pe' will be relatively 
                                       ## fast copared to having a GPU have to get the data from a CPU.

    def forward(self, token_embeddings):
        """ 
        NOTE here '.unseueeze(1)' is used for translation data since token_embedding = (seq, batch, embd_dim)
        and sel.pe has dimension of (max_len, embd_dim). 
        We need (max_len, embd_dim) -> (max_len, 1, embd_dim) and '+' token_embeddings
        """
        return self.dropout(token_embeddings + self.pe[:token_embeddings.size(0), :].unsqueeze(1))



# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


if __name__ == '__main__':
    word_embedding = torch.tensor([[ 0.3367,  0.1288], [ 0.2345,  0.2303], [-1.1229, -0.1863], [ 0.4617,  0.2674]])
    print(word_embedding.size(1))
    pos_encoding = PositionalEncoding(word_embedding.shape[1], max_len=6, dropout=0.1)
    print(pos_encoding(word_embedding))
