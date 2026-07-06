# Transformers for Time-Series Forecasting

Time-series has sequential data, and can be obviously implemented with transformers.

Followed the github repo [Time-Series-Forecasting](https://github.com/rezaAdinepour/Time-Series-Forecasting), we build a transformer time-series forecasting model. Even just one layer, the results look to have very good performance. Refer the notebook [here](https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/Sequential-Data/Time-Series-Forecast/simple_ts_transformer/train_ts_forecasting.ipynb).

By the medium article: [Transformers for Time Series Forecasting](https://medium.com/@serana.ai/transformers-for-time-series-forecasting-e5e0327e78be), there are more complicated transformer architectures for the time-series forecasting directions: 
* Informer
* Autoformer
* Temporal Fusion Transformer
* FEDformer
* iTransformer
* PatchTST



## Informer


<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/Sequential-Data/Time-Series-Forecast/images/informer.png" width="600">

(The image is from the paper Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (2021) by Zhou et al.)

One of Informer’s key features is the **ProbSparse attention** mechanism, which speeds up the model by reducing the computation needed. Instead of comparing every part of the input to every other part (which takes time that grows quadratically as the input gets longer), it focuses only on the most important parts.

Another key feature is **self-attention distillation**, where the sequence length is halved at each layer.

Informer also outputs the entire forecast in a single forward pass rather than predicting step-by-step. It may also face challenges when the data is extremely noisy or behaves unpredictably over long periods of the historical data.

Reference:
* Zhou et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (2021)
* [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://rezayazdanfar.medium.com/informer-beyond-efficient-transformer-for-long-sequence-time-series-forecasting-4eeabb669eb)

## Autoformer

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/Sequential-Data/Time-Series-Forecast/images/autoformer.png" width="700">

(The image is from the paper DMEformer: A newly designed dynamic model ensemble transformer for crude oil futures prediction, (2023) by Liu et al.)

Unlike models that decompose data as a pre-processing step, Autoformers perform decomposition inside the model **during** both training and inference. This allows it to **adaptively learn** patterns as part of the forecasting task.

First feature for autoformer is **series decomposition**. Instead of processing the raw signal directly, Autoformer separates it into **trend** and **seasonal** components.

Second feature is Auto-Correlation attention, which compares similar sub-sequences instead of every time step with each other. For example, it learns to compare Mondays with other Mondays.

Autoformer is a deterministic model, meaning it outputs a single forecast rather than a distribution. 

Reference:
* Liu et al. DMEformer: A newly designed dynamic model ensemble transformer for crude oil futures prediction, (2023)


## Temporal Fusion Transformer (TFT)

TFT is a transformer-based model designed for multi-horizon time-series forecasting, with a focus on interpretability and handling complex, real-world data. 

Traditional models often fall short when dealing with mixed inputs like static features, known future events, and dynamic variables. TFT addresses these challenges using a combination of gating mechanisms, attention layers, and variable selection networks to model both short and long-term dependencies.