# Diffusion Model

The note here follows Prof. Hung-Yi Lee's class. See [淺談圖像生成模型 Diffusion Model 原理](https://www.youtube.com/watch?v=azBugJzmz-o)


## Reverse Process

The image generation is a sequence of denoising processes starting from random sample and keep removing noises.

This is called **reverse process**.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/reverse_process_1.png" width="900">

The denoise module is to predict noise, given by noised images and the step as input 

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/denoise_1.png" width="700">

The denoised image is the input noised images removing the predicted noise.


## Forward Process


To train the denoise predictor, we need to prepare the paired data: input features are noised images and the step, and the label (ground truth) are noise.

The process to add (Gaussian) noise to images is called **forward process**.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/forward_process_1.png" width="800">



