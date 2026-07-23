# Diffusion Model

## Reverse Process

The image generation is a sequence of denoising processes starting from random sample + + removing predicted noises.

This is called **reverse process**.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/reverse_process_1.png" width="900">

The denoise module is to predict noise given noised input and the step

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/denoise_1.png" width="700">

Therefore, we need to prepare the paired data to train the denoise predictor, where input data are noised input and the step, and the label are noise.

## Forward Process

The process to add noise to images is called **forward process**.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/forward_process_1.png" width="700">



