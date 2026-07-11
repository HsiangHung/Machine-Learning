import pandas as pd

RATINGS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
BOOKS_URL   = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"


def get_data(url, csvname):
    df = pd.read_csv(url)
    # To save a local copy
    print(df.shape)
    df.to_csv(f"{csvname}.csv", index=False)


get_data(RATINGS_URL, "ratings")
get_data(BOOKS_URL, "books")
