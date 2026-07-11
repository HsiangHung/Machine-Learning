import torch
import pandas as pd
import numpy as np

# -----------------------------
# Choose device: MPS (Mac), CUDA, or CPU
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def get_raw_data():
    print("Downloading ratings...")
    # ratings = pd.read_csv("/content/drive/MyDrive/data/Two-Tower/ratings.csv")
    ratings = pd.read_csv("./ratings.csv")

    print("Downloading books...")
    # books = pd.read_csv("/content/drive/MyDrive/data/Two-Tower/books.csv")
    books = pd.read_csv("./books.csv")

    # Quick peek at data shape + a few rows
    print("ratings:", ratings.shape)
    print("books:", books.shape)
    return ratings, books


def process_rating(ratings):
    ratings["label"] = ratings.apply(lambda x: 1 if x["rating"] >= 4 else 0, axis=1)
    ratings["label"] = ratings["label"].astype("int8")
    return ratings


def process_book_feature(books):
    """
    process columns:
        * year_bucket: decade bucket
        * language_code: language string (fill missing as "UNK")
        * primary_author: first author name
    """
    book_features = books[["book_id", "original_publication_year", "language_code", "authors"]].copy()
    print(book_features.head())

    # ====================================
    #  YEAR
    # Ensure publication year is numeric; non-numeric becomes NaN
    book_features["original_publication_year"] = pd.to_numeric(book_features["original_publication_year"], errors="coerce")

    # and then Bucket year by decade: e.g., 1999 -> 1990, 2003 -> 2000
    book_features["year_bucket"] = ((book_features["original_publication_year"] // 10) * 10)

    # ====================================
    # LANGUAGE
    book_features["language_code"] = (
        book_features["language_code"]
        .fillna("UNK")
        .astype(str)
        .str.strip()
    )
    book_features.loc[book_features["language_code"] == "", "language_code"] = "UNK"


    # Extract a single "primary author" by taking the first in the comma-separated list
    book_features["primary_author"] = (
        book_features["authors"]
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )

    # build author vocabulary
    book_features["primary_author"] = (
        book_features["primary_author"]
        .fillna("UNK")
        .astype(str)
    )

    # ====================================
    # PRIMARY AUTHOR
    # book_features["primary_author"] = book_features.apply(lambda x: x["authors"].strip().split(",")[0], axis=1)
    return book_features


def encode_features(df):
    """
     Encode id features
    """
    # e.g. encode language to integer ids (0..num_langs-1)
    df["user_id"] = df["user_id"].astype("category").cat.codes.astype(np.int64)
    df["item_id"] = df["book_id"].astype("category").cat.codes.astype(np.int64)
    df["lang_id"] = df["language_code"].astype("category").cat.codes.astype(np.int64)
    df["year_id"] = df["year_bucket"].astype("category").cat.codes.astype(np.int64)

    author_to_id = {a: i for i, a in enumerate(df["primary_author"].unique())}
    df["author_id"] = df["primary_author"].map(author_to_id).astype(np.int64)

    return df


def active_filter(ratings):
    user_counts = ratings["user_id"].value_counts()
    # Keep users with between 10 and 1000 ratings for a denser, more stable toy dataset
    active_users = user_counts[(user_counts >= 10) & (user_counts <= 1000)].index
    # Filter the ratings table down to those users
    mini = ratings[ratings["user_id"].isin(active_users)].copy()
    
    # Count ratings per book within the filtered user subset
    book_counts = mini["book_id"].value_counts()
    # Keep books with at least 50 interactions in this subset
    popular_books = book_counts[book_counts >= 50].index

    # Filter again
    mini = mini[mini["book_id"].isin(popular_books)].copy()

    print("mini:", len(mini))

    return mini


def get_data_df(data_subset=False):
    ratings, books = get_raw_data()

    ratings = process_rating(ratings)
    if data_subset:
        ratings = active_filter(ratings)

    book_features = process_book_feature(books)

    join_df = ratings.merge(book_features, on="book_id", how="left")

    # fill missing year_bucket
    join_df["year_bucket"] = join_df["year_bucket"].fillna(-1).astype(np.int32)
    join_df = encode_features(join_df)

    print(join_df.head())
    return join_df


if __name__ == "__main__":
    get_data_df(data_subset=False)
