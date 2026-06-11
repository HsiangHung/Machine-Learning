# Batching

LLMs often fail to fully utilize these GPUs because much of the chip's memory bandwidth is spent loading model parameters. Batching helps mitigate this bottleneck. In production, your service might be flooded with multiple requests arriving at the same time. Instead of processing each request individually, batching them together allows you to use the same loaded model parameters across multiple requests, thus dramatically improving throughput.

## Static batching

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Optimizaton/images/static_batching.png" width="600">


* The first request in a batch is forced to wait for the last one, adding unnecessary delay. Picture a printer that won’t start printing until you’ve queued up a set number of documents, regardless of how long it takes for the last document to arrive.

* Not all requests in a batch are created equal. In LLM inference, some requests may generate very short responses, while others could involve lengthy, step-by-step reasoning. Since all requests in the batch must wait until the slowest one finishes, this can lead to wasted compute resources and increased latency.