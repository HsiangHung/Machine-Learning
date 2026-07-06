# Transformers for Time Series Forecasting

Six key architectures that represent distinct directions in the field: 
* Informer
* Autoformer
* FEDformer
* iTransformer
* Temporal Fusion Transformer (TFT)
* PatchTST


## Autoformer

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Optimizaton/images/dynamic_batching.png" width="700">

Unlike models that decompose data as a pre-processing step, Autoformers perform decomposition inside the model **during** both training and inference. This allows it to **adaptively learn** patterns as part of the forecasting task.

First feature for autoformer is **series decomposition**. Instead of processing the raw signal directly, Autoformer separates it into **trend** and **seasonal** components.

Second feature is Auto-Correlation attention, which compares similar sub-sequences instead of every time step with each other. For example, it learns to compare Mondays with other Mondays.


[Transformers for Time Series Forecasting](https://medium.com/@serana.ai/transformers-for-time-series-forecasting-e5e0327e78be)