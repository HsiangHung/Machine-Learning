"""
 data source: https://www.microsoft.com/en-us/research/project/mslr/
"""
import pandas as pd


def parse_txt_to_df(data_dir: str = "./MSLR-WEB10K/Fold1", filename: str = "train.txt") -> pd.DataFrame:
    with open(f"{data_dir}/{filename}", "r") as file:
        data = []
        # i = 0
        for line in file:
            row = line.strip().split(" ")
            data.append([int(row[0])]+ [float(x.split(":")[1]) for x in row[1:]]) # relevance + qid + features...
            # print(line.strip())
            # i += 1
            # if i==6:
                # break
        df = pd.DataFrame(data, columns=["relevance", "qid"] + [f"col-{i}" for i in range(1, 137)])

    return df


def get_data(folder_index=1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    read folder individually and concat together
    """
    m = folder_index
    train_df = parse_txt_to_df(data_dir=f"./MSLR-WEB10K/Fold{m}", filename="train.txt")            
    val_df = parse_txt_to_df(data_dir=f"./MSLR-WEB10K/Fold{m}", filename="vali.txt")
    test_df = parse_txt_to_df(data_dir=f"./MSLR-WEB10K/Fold{m}", filename="test.txt")
    return train_df, val_df, test_df
