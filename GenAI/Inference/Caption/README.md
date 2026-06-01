# Caption Inference Using VLM

Use VLM model (`SmolVLM2`) to generate captions given images.

There are three parts:
* Single Input Caption, i.e. each input has one image.
* Batch Input Caption, i.e. each input is a batch, with multiple images. Here batch_size=16.
* Batch Captioning on Multiple GPUs, i.e. each input is a batch and use multiGPU in parallel.