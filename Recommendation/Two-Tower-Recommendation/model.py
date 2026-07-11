import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def split_train_val_test(df):
    seed = 42

    # Create training set (80% of data)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )

    # Create validation and testing set from equal halves of the remaining 20% of data
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )

    print(f"training size: {len(train_df)}, val size: {len(val_df)}, test size: {len(test_df)}")
    return train_df, val_df, test_df


class TwoTowerModel(nn.Module):
    """
    Two-tower recommender with:
      - User ID embedding
      - Item ID embedding
      - Item metadata embeddings (language, year bucket, author)
      - User/item bias terms
      - Scaled dot-product scoring
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_langs: int,
        num_year_buckets: int,
        num_authors: int,
        emb_dim: int = 64,
        lang_emb_dim: int = 4,
        year_emb_dim: int = 4,
        author_emb_dim: int = 8,
    ):
        super().__init__()

        # -----------------------------
        # Core ID embeddings
        # -----------------------------
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        # -----------------------------
        # Bias terms (scalars) for popularity/activity effects
        # -----------------------------
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # -----------------------------
        # Metadata embeddings
        # -----------------------------
        self.lang_emb = nn.Embedding(num_langs, lang_emb_dim)
        self.year_emb = nn.Embedding(num_year_buckets, year_emb_dim)
        self.author_emb = nn.Embedding(num_authors, author_emb_dim)

        # -----------------------------
        # Project metadata -> emb_dim
        # -----------------------------
        # Concatenate [lang_emb, year_emb, author_emb] -> MLP -> emb_dim
        self.item_feat = nn.Sequential(
            nn.Linear(lang_emb_dim + year_emb_dim + author_emb_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )

        # Scale factor for dot product (stabilizes logits)
        self.scale = math.sqrt(emb_dim)

        # Initialize biases to zero (common practice)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    # -----------------------------
    # Encoders
    # -----------------------------
    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Return user embeddings: shape [B, D]."""
        return self.user_emb(user_ids)

    def encode_item(
        self,
        item_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        year_ids: torch.Tensor,
        author_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return item embeddings enriched with metadata:
          v = v_id + f(metadata)
        """

        # Item id embedding
        v_id = self.item_emb(item_ids)  # [B, D]

        # Metadata embeddings
        v_lang = self.lang_emb(lang_ids)        # [B, L]
        v_year = self.year_emb(year_ids)        # [B, Y]
        v_author = self.author_emb(author_ids)  # [B, A]

        # Concatenate metadata vectors
        x = torch.cat([v_lang, v_year, v_author], dim=-1)  # [B, L+Y+A]

        # Project metadata into item embedding space
        v_feat = self.item_feat(x)  # [B, D]

        # Combine memorization (id) + generalization (metadata)
        return v_id + v_feat

    # -----------------------------
    # Scoring forward pass
    # -----------------------------
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        year_ids: torch.Tensor,
        author_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores for (user, item) pairs.
        Supports broadcasting shapes if user_ids/item_ids are broadcastable.
        """

        # Get user and item representations
        u = self.encode_user(user_ids)
        v = self.encode_item(item_ids, lang_ids, year_ids, author_ids)

        # Scaled dot product similarity
        dot = (u * v).sum(dim=-1) / self.scale

        # Add user/item biases
        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.item_bias(item_ids).squeeze(-1)

        return dot + bu + bi


if __name__ == "__main__":

    num_users = 53424
    num_items = 10000
    num_langs = 26     # num of languages
    num_year_buckets = 84
    num_authors = 3888

    # -----------------------------
    # Initialize model + optimizer
    # -----------------------------
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_langs=num_langs,
        num_year_buckets=num_year_buckets,
        num_authors=num_authors,
        emb_dim=64,
    ).to(device)

    print(f"num_users: {num_users}, num_items: {num_items}, num_langs: {num_langs}, num_year_buckets: {num_year_buckets}, num_authors: {num_authors}")

    # Adam with mild weight decay (L2 regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)