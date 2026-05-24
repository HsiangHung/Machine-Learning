import pandas as pd
import numpy as np
from sklearn.metrics import (
    ndcg_score,
    average_precision_score,
)
from scipy.stats import rankdata


# NDCG
def test_ndcg_score(y_test, pred):
    true_relevance = y_test.sort_values(ascending=False)

    # pred = model.predict(X_test)
    res_df = pd.DataFrame({"true_rel": y_test, "pred_rel": pred})
    relevance_score = res_df.sort_values("pred_rel", ascending=False)

    print(res_df.sort_values("pred_rel", ascending=False).head(50))

    # Use computed variables to calculate the nDCG score
    print(
        "nDCG score: ",
        round(ndcg_score(
            [true_relevance.to_numpy()], [relevance_score["true_rel"].to_numpy()]
        ), 4),
    )


# MRR (mean_reciprocal_rank)
def test_mrr_score(y_test, pred):
    mrrs = []
    for true_labels, pred_scores in zip(y_test, pred):
        # Rank predictions in descending order (highest score = rank 1)
        ranks = rankdata(-pred_scores, method='ordinal')
        
        # Get the rank of the true label (assuming one target, using argmax)
        true_rank = ranks[np.argmax(true_labels)]
        
        # Calculate reciprocal rank
        mrrs.append(1.0 / true_rank)
    
    print(f"MRR score: {round(np.mean(mrrs), 4)}")
    return np.mean(mrrs)


# average_precision_score
def test_map_score(y_test, pred):
    # print(type(y_test), np.array(y_test.values).reshape(-1, 1).shape)
    # print(type(pred), pred.reshape(-1, 1).shape)
    map_score = average_precision_score(
        np.array(y_test.values).reshape(-1, 1),
        pred.reshape(-1, 1)
    )
    print(f"MAP score: {round(map_score, 4)}")


def test_pipeline(model, test_df):

    X_test = test_df.drop(["relevance", "qid"], axis=1)
    y_test = test_df["relevance"]
    pred = model.predict(X_test)

    test_ndcg_score(y_test, pred)
    test_mrr_score(y_test, pred)
    test_map_score(y_test, pred)
