"""
LoghtGBM LambdaMART training example:
* https://github.com/lezzhov/learning_to_rank/blob/main/learning_to_rank/scripts/train.py
* https://medium.datadriveninvestor.com/a-practical-guide-to-lambdamart-in-lightgbm-f16a57864f6
* https://tamaracucumides.medium.com/learning-to-rank-with-lightgbm-code-example-in-python-843bd7b44574
"""
import pandas as pd
import numpy as np

import lightgbm as lgb

from data import get_data
from eval import (
    test_mrr_score,
    test_map_score,
    test_ndcg_score,
    test_pipeline
)


lgbm_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "seed": 42,
    "num_leaves": 15,
    "learning_rate": 0.1,
    "reg_lambda": 2.5,
    "verbose": -1,
    "min_data_in_leaf": 1, # LGBM in default leaf > 20, so need this config to keep run
    "min_sum_hessian_in_leaf": 0.0,
    "saved_model_path": "./saved_lgbm",
}


def main():

    train_df, val_df, test_df = get_data(folder_index=1)

    # train_df = train_df[:500]
    # val_df = val_df[:500]

    print(f"train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")
    print(train_df.head())

    X_train = train_df.drop(["relevance", "qid"], axis=1)
    y_train = train_df["relevance"]

    X_val = val_df.drop(["relevance", "qid"], axis=1)
    y_val = val_df["relevance"]

    # ---- group for train and val set ------------------
    querys = train_df["qid"].tolist()
    group_train = [querys.count(q) for q in sorted(set(querys))]

    querys = val_df["qid"].tolist()
    group_val = [querys.count(q) for q in sorted(set(querys))]
    #print(group_val)

    # ------ define lgbm model -----------
    model = lgb.LGBMRanker(
        n_estimators=10000,
        objective=lgbm_params["objective"],
        metric=lgbm_params["metric"],
        num_leaves=lgbm_params["num_leaves"],
        learning_rate=lgbm_params["learning_rate"],
        # min_data_in_leaf=lgbm_params["min_data_in_leaf"],
        reg_lambda=lgbm_params["reg_lambda"],
        verbose=-1,
    )

    model.fit(
        X=X_train,
        y=y_train,
        group=group_train,
        eval_group=[group_val],
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150)
        ],
    )

    model.booster_.save_model(lgbm_params["saved_model_path"], num_iteration=model._best_iteration)

    # X_test = test_df.drop(["relevance", "qid"], axis=1)
    # y_test = test_df["relevance"]
    # pred = model.predict(X_test)
    test_pipeline(model, test_df)

    lgb.plot_importance(model, figsize = (12,8))


if __name__ == "__main__":
    main()