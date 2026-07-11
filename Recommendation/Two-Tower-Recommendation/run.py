"""
Reference:

Grokking Two-Tower Models
https://medium.com/@cole.ian.diamond/grokking-two-tower-models-53e0140897e2

"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data import get_data_df
from dataloader import get_dataloader
from eval import (
    recall_at_k_mask_seen,
    recall_at_k_pop_mask_seen,
)
from model import (
    split_train_val_test,
    TwoTowerModel,
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def train(
    model,
    dataloader,
    optimizer,
    device,
    item_lang_ids,      # [num_items]
    item_year_ids,      # [num_items]
    item_author_ids,    # [num_items]
    info_nce=True,
    temperature=1.0,
):
    """
    Train for one epoch.

    For each batch:
      - compute s_pos for (u, pos_item)
      - compute s_neg for (u, neg_items)
      - InfoNCE: cross-entropy over [s_pos, s_neg...]
    """
    model.train()

    total_loss = 0.0
    steps = 0

    # Move item metadata tensors to device once
    item_lang_ids = item_lang_ids.to(device)
    item_year_ids = item_year_ids.to(device)
    item_author_ids = item_author_ids.to(device)

    for batch in dataloader:
        # Batch tensors
        u = batch["user_id"].to(device)           # [B]
        i_pos = batch["pos_item_id"].to(device)   # [B]
        i_neg = batch["neg_item_id"].to(device)   # [B, Nneg]

        # Lookup metadata ids for positive items
        pos_lang   = item_lang_ids[i_pos]         # [B]
        pos_year   = item_year_ids[i_pos]         # [B]
        pos_author = item_author_ids[i_pos]       # [B]

        # Lookup metadata ids for negative items
        neg_lang   = item_lang_ids[i_neg]         # [B, Nneg]
        neg_year   = item_year_ids[i_neg]         # [B, Nneg]
        neg_author = item_author_ids[i_neg]       # [B, Nneg]

        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Positive scores s(u, i_pos): [B]
        s_pos = model(u, i_pos, pos_lang, pos_year, pos_author)

        # Expand users to match the negative item tensor shape
        # u_neg: [B, Nneg]
        u_neg = u.unsqueeze(1).expand_as(i_neg)

        # Negative scores s(u, i_neg): [B, Nneg]
        s_neg = model(u_neg, i_neg, neg_lang, neg_year, neg_author)

        # InfoNCE / sampled softmax:
        # candidates scores: [B, 1 + Nneg], label is 0 (positive at index 0)
        scores = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1)  # [B, 1+Nneg]
        scores = scores / temperature                           # optional sharpening
        targets = torch.zeros(scores.size(0), dtype=torch.long, device=device)
        loss = F.cross_entropy(scores, targets)

        # print(loss)

        # Backprop + optimizer step
        loss.backward()
        optimizer.step()

        # total_loss += float(loss.item())
        total_loss += loss.item()
        steps += 1
        # return

    return total_loss / max(steps, 1)


def popularity_val(model_r20,
                   num_items,
                   item_lang_ids,
                   item_year_ids,
                   item_author_ids,
                   train_seen,
                   eval_pos, 
                   train_df,
):

    # -----------------------------
    # Popularity list from TRAIN positives (global)
    # -----------------------------
    popular_items = (
        train_df[train_df["label"] == 1]
        .groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # -----------------------------
    # Compute both metrics on SAME users (cap optional)
    # -----------------------------
    MAX_USERS = 2000  # set None for all users (slower but best)

    pop_r20 = recall_at_k_pop_mask_seen(
        eval_pos=eval_pos,
        train_seen=train_seen,
        popular_items=popular_items,
        K=20,
        max_users=MAX_USERS,
    )

    print(f"Masked Recall@20 (Model): {model_r20:.4f}")
    print(f"Masked Recall@20 (Pop)  : {pop_r20:.4f}")
    print(f"Lift (abs): {model_r20 - pop_r20:+.4f}")
    print(f"Lift (rel): {(model_r20 / pop_r20 - 1.0)*100:+.2f}%")


def main():
    df = get_data_df(data_subset=True)
    train_df, val_df, test_df = split_train_val_test(df)

    print(f"training size: {len(train_df)}, val size: {len(val_df)}, test size: {len(test_df)}")

    # -----------------------------
    # Build item-level metadata tensors aligned to item_idx
    # -----------------------------
    # We need a single row per item_idx containing its lang/year/author indices.
    item_meta = (
        df[["item_id", "lang_id", "year_id", "author_id"]]
        .drop_duplicates("item_id")
        .sort_values("item_id")
    )
    print(item_meta.shape)

    # -------------------------------------------------------------------------------
    # Convert metadata columns into torch tensors of shape [num_items]
    item_lang_ids   = torch.tensor(item_meta["lang_id"].values, dtype=torch.long)
    item_year_ids   = torch.tensor(item_meta["year_id"].values, dtype=torch.long)
    item_author_ids = torch.tensor(item_meta["author_id"].values, dtype=torch.long)

    # Cardinalities
    num_users = int(df["user_id"].nunique())
    num_items = int(df["item_id"].nunique())
    num_langs = int(df["lang_id"].nunique())
    num_year_buckets = int(df["year_id"].nunique())
    num_authors = int(df["author_id"].nunique())

    # Sanity checks: item_meta should be exactly one row per item, and ordered 0..num_items-1
    assert len(item_meta) == num_items
    assert (item_meta["item_id"].values == torch.arange(num_items).numpy()).all()

    # ------------------------------
    #   dataloader 
    train_loader = get_dataloader(num_items, train_df)
    print()
    print(" ** dataloader is ready ** ")
    print()

    # -----------------------------
    # Build train_seen (mask set) and eval_pos (held-out positives)
    # -----------------------------
    train_seen = (
        train_df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    # Evaluate on validation set during training
    eval_df = val_df

    # Held-out positives per user in eval split
    eval_pos = (
        eval_df[eval_df["label"] == 1]
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    print(f"train users: {len(train_seen)}, eval users: {len(eval_pos)}")

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
    print()
    print(" ** model is ready ** ")

    # -----------------------------
    # Train loop
    # -----------------------------
    num_epochs = 10
    losses = []
    recall20s = []
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        loss = train(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            item_lang_ids=item_lang_ids,
            item_year_ids=item_year_ids,
            item_author_ids=item_author_ids,
            info_nce=True,         # sampled softmax
            temperature=1.0,       # can tune
        )

        # Evaluate masked recall@20
        r20 = recall_at_k_mask_seen(
            model=model,
            eval_pos=eval_pos,
            train_seen=train_seen,
            num_items=num_items,
            item_lang_ids=item_lang_ids,
            item_year_ids=item_year_ids,
            item_author_ids=item_author_ids,
            K=20,
            device=device,
            max_users=2000,        # cap for speed; set None for full eval
        )

        losses.append(loss)
        recall20s.append(r20)
        print(f"Epoch {epoch:02d} | loss: {loss:.4f} | masked recall@20: {r20:.4f}")

    popularity_val(r20, num_items, item_lang_ids, item_year_ids, item_author_ids, train_seen, eval_pos, train_df)

if __name__ == "__main__":
    main()
