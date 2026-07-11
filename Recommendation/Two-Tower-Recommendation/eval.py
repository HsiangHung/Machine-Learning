import torch

@torch.no_grad()
def recall_at_k_mask_seen(
    model,
    eval_pos,            # dict: user -> set of held-out positive items
    train_seen,          # dict: user -> set of training-seen items to mask
    num_items,
    item_lang_ids,       # torch tensor of shape [num_items]
    item_year_ids,       # torch tensor of shape [num_items]
    item_author_ids,     # torch tensor of shape [num_items]
    K=20,
    device="cpu",
    max_users=None,
):
    """
    Compute average per-user Recall@K with masking of items seen in training.

    Steps per user:
      - Score all items
      - Mask train_seen items (set their scores to -inf)
      - Take top-K
      - Recall = hits_in_topK / num_heldout_positives
    """
    model.eval()

    # Move item metadata tensors to device once
    item_lang_ids = item_lang_ids.to(device)
    item_year_ids = item_year_ids.to(device)
    item_author_ids = item_author_ids.to(device)

    # All item ids [0..num_items-1]
    all_item_ids = torch.arange(num_items, device=device, dtype=torch.long)

    # Precompute all item embeddings and biases (fast for repeated user scoring)
    all_item_embs = model.encode_item(
        all_item_ids,
        item_lang_ids,
        item_year_ids,
        item_author_ids
    )  # [I, D]
    all_item_bias = model.item_bias(all_item_ids).squeeze(-1)  # [I]

    # User list to evaluate
    users = list(eval_pos.keys())
    if max_users is not None:
        users = users[:max_users]

    recalls = []

    for u in users:
        # Held-out positives for this user
        pos_set = eval_pos.get(u, set())
        if not pos_set:
            continue

        # Compute user embedding + bias
        user_id = torch.tensor([u], device=device, dtype=torch.long)
        user_emb = model.encode_user(user_id)                # [1, D]
        user_b   = model.user_bias(user_id).squeeze(-1)      # [1]

        # Dot-product scores for all items: [I]
        scores = (user_emb @ all_item_embs.T).squeeze(0) / model.scale

        # Add item biases and user bias
        scores = scores + all_item_bias + user_b

        # Mask training-seen items so we don’t “recommend” what they already consumed
        seen = train_seen.get(u, set())
        if seen:
            seen_idx = torch.tensor(list(seen), device=device, dtype=torch.long)
            scores[seen_idx] = -1e9

        # Top-K predicted item ids
        topk = torch.topk(scores, K, largest=True).indices.tolist()

        # Count hits among held-out positives
        hits = sum((i in pos_set) for i in topk)
        recalls.append(hits / len(pos_set))

    # Average recall across users
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def recall_at_k_pop_mask_seen(eval_pos, train_seen, popular_items, K=20, max_users=None):
    users = list(eval_pos.keys())
    if max_users is not None:
        users = users[:max_users]

    recalls = []
    for u in users:
        pos_set = eval_pos.get(u, set())
        if not pos_set:
            continue

        seen = train_seen.get(u, set())

        recs = []
        for it in popular_items:
            if it not in seen:
                recs.append(it)
            if len(recs) >= K:
                break

        hits = sum((i in pos_set) for i in recs)
        recalls.append(hits / len(pos_set))

    return float(sum(recalls) / len(recalls)) if recalls else 0.0
