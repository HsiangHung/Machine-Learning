# Batching

Reference: [Bento ML: Static, dynamic and continuous batching](https://bentoml.com/llm/inference-optimization/static-dynamic-continuous-batching)

Though GPUs are designed for highly parallel computation workloads, LLMs often fail to fully utilize these GPUs because much of the chip's memory bandwidth is spent loading model parameters. Batching helps mitigate this bottleneck. In production, your service might be flooded with multiple requests arriving at the same time. 

Instead of processing each request individually, batching them together allows you to use the same loaded model parameters across multiple requests, thus dramatically improving throughput.

## Static batching

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Optimizaton/images/static_batching.png" width="700">


* The first request in a batch is forced to wait for the last one, adding unnecessary delay. Picture a printer that won’t start printing until you’ve queued up a set number of documents, regardless of how long it takes for the last document to arrive.

* Not all requests in a batch are created equal. In LLM inference, some requests may generate very short responses, while others could involve lengthy, step-by-step reasoning. Since all requests in the batch must wait until the slowest one finishes, this can lead to wasted compute resources and increased latency.


## Dynamic batching

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Optimizaton/images/dynamics_batching.png" width="700">

* Dynamic batching collects incoming requests into batches, but it doesn’t insist on a fixed batch size. Instead, it sets a time window and processes whatever requests have arrived in that time frame. If the batch reaches its size limit sooner, it launches immediately.
* From above picture, some batches might not be completely full when launched, it doesn’t always achieve maximum GPU efficiency.
* Another drawback is that, like static batching, the longest request in a batch still dictates when the batch finishes; short requests have to wait unnecessarily.


## Continuous batching

For LLM inference, output sequences vary widely in length. Some users might ask simple questions, while others request detailed explanations. Static and dynamic batching force the short requests to wait for the longest one. This leaves GPU resources unsaturated.


<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Optimizaton/images/continuous_batching.png" width="700">


* Continuous batching (aka in-flight batching), addresses these inefficiencies. Continuous batching doesn’t force the entire batch to complete before returning results. Instead, it lets each sequence in a batch finish independently and immediately replaces it with a new one. 
* This technique uses iteration-level scheduling, meaning the batch composition changes dynamically at each decoding iteration. 
* As soon as a sequence in the batch finishes generating tokens, the server inserts a new request in its place. 

This maximizes GPU occupancy and keeps compute resources busy by avoiding idle time that would otherwise be spent waiting for the slowest sequence in a batch to finish.

Major inference frameworks such as vLLM, SGLang, TensorRT-LLM (in-flight batching) and LMDeploy (persistent batching) all support continuous batching or similar mechanisms. For memory management in long or mixed-length batches, see PagedAttention.

