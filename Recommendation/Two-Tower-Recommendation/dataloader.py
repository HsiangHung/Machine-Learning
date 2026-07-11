import torch
from torch.utils.data import Dataset, DataLoader
import random


class GoodreadsData(Dataset):
    """
    Yields (user, positive_item, negative_items) for two-tower training.

    We expect df_pos to contain ONLY positive interactions.
    Negative items are sampled uniformly from all items, excluding items the user liked.
    """
    def __init__(
        self,
        df_pos,
        user_to_pos_set,
        num_items,
        num_neg=5,
        seed=42,
    ):
        # Store arrays of users and positive items for indexing
        self.users = df_pos["user_id"].values
        self.pos_items = df_pos["item_id"].values

        # Dict: user -> set of positive items (to avoid sampling positives as negatives)
        self.user_to_pos = user_to_pos_set

        # Total item universe size
        self.num_items = int(num_items)

        # How many negatives to sample per positive
        self.num_neg = int(num_neg)

        # RNG used for negative sampling (deterministic given seed)
        self.rng = random.Random(seed)

    def __len__(self):
        # Number of positive examples
        return len(self.users)

    def __getitem__(self, idx):
        # Grab one positive training row
        u = int(self.users[idx])
        i_pos = int(self.pos_items[idx])

        # User’s known positives (to avoid)
        pos_set = self.user_to_pos.get(u, set())

        # Sample negatives
        negs = []
        tries = 0

        # In rare cases, a user might have liked a huge fraction of items
        # making it hard to find unseen items. 
        # The tries limit prevents an infinite loop.
        while len(negs) < self.num_neg and tries < 1000:
            # Sample random item id
            j = self.rng.randrange(self.num_items)

            # Ensure it’s not a known positive and not the same as i_pos
            if j not in pos_set and j != i_pos:
                negs.append(j)

            tries += 1

        # Fallback if user has too many positives (rare in our filtered dataset)
        if len(negs) < self.num_neg:
            while len(negs) < self.num_neg:
                negs.append(self.rng.randrange(self.num_items))

        # Return tensors (DataLoader will batch them)
        return {
            "user_id": torch.tensor(u, dtype=torch.long),
            "pos_item_id": torch.tensor(i_pos, dtype=torch.long),
            "neg_item_id": torch.tensor(negs, dtype=torch.long),  # shape: [num_neg]
        }


def get_dataloader(num_items, train_df, batch_size=1024):
    # -----------------------------
    # Build training positives (ratings >= 4) and user -> pos lookup
    # -----------------------------
    train_pos_df = train_df[train_df["label"] == 1].copy()
    print(train_df.shape, train_pos_df.shape)

    # Map each user to the set of items they liked in training
    user_to_train_pos = (
        train_pos_df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    # -----------------------------
    # Dataloader
    train_dataset = GoodreadsData(
        df_pos=train_pos_df,
        user_to_pos_set=user_to_train_pos,
        num_items=num_items,
        num_neg=10,   # negative sampling, typically 5–20
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # stable batch shapes
        num_workers=0,   # use 0 for notebook simplicity; increase for speed
    )

    # Sanity check a batch
    batch = next(iter(train_loader))
    print({k: v.shape for k, v in batch.items()})
    print({k: v.dtype for k, v in batch.items()})
    return train_loader
