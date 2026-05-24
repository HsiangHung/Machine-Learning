
# LightGBM-Ranker

This folder has the tutoral notebook (tutorial-lgbm-ranker.ipynb) to train a lightGBM ranker model.

NOTE:
* The objective to train the LightGBM ranker is **Lambdarank**.
* The dataset used to train the ranker is released from **MSLR-WEB10K** by Miscrosoft: https://www.microsoft.com/en-us/research/project/mslr/.
* The LoghtGBM LambdaMART training example codes:
    * https://github.com/lezzhov/learning_to_rank/blob/main/learning_to_rank/scripts/train.py
    * https://medium.datadriveninvestor.com/a-practical-guide-to-lambdamart-in-lightgbm-f16a57864f6
* To train the model, `python run.py`
* Even 1M dataset size, training the lightGBM is pretty fast; 10mins order.

## Reference


* [Learning-to-rank with LightGBM (Code example in python)]: https://medium.com/@tacucumides/learning-to-rank-with-lightgbm-code-example-in-python-843bd7b44574
[[Tamara Alexandra Cucumides] Learning-to-rank with LightGBM (Code example in python)](https://medium.com/@tacucumides/learning-to-rank-with-lightgbm-code-example-in-python-843bd7b44574)

* [A Practical Guide to LambdaMART in LightGbm]: https://medium.datadriveninvestor.com/a-practical-guide-to-lambdamart-in-lightgbm-f16a57864f6
[[Akash Dubey] A Practical Guide to LambdaMART in LightGbm](https://medium.datadriveninvestor.com/a-practical-guide-to-lambdamart-in-lightgbm-f16a57864f6)
