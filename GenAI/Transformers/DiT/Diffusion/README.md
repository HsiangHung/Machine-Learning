# Diffusion Model

The note here follows Prof. Hung-Yi Lee's class. See [淺談圖像生成模型 Diffusion Model 原理](https://www.youtube.com/watch?v=azBugJzmz-o)



## Reverse Process

The image generation by diffusion models is a sequence of denoising processes starting from random sample and keep removing noises until a good image.

This is called **reverse process**.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/reverse_process_1.png" width="800">

The denoise module is to predict noise, given by noised images and the step as input 

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/denoise_1.png" width="600">

The output denoised image is the input noised images removing the predicted noise.


## Forward Process


To train the **noise predictor**, we need to prepare the paired data: 
* Features: noised images and the step
* Label (ground truth): noise.

The process to add (Gaussian) noise to images is called **forward (diffusion) process**.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/forward_process_1.png" width="800">


## Text-Image 

For text-image diffusion model, the text will be additional input:

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/denoise_2.png" width="700">

Now in the forward process, we have three input:
* Features: noised images, the step and text
* Label (ground truth): noise.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/forward_process_2.png" width="800">


## Diffusion Model Theory

Prof. Lee also have following lectures [Diffusion Model 原理剖析 (1/4) (optional)](https://www.youtube.com/watch?v=ifCDXFdeaaM) to mention that VAE and diffusion are actually very similar.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/VAE_vs_diffusion.png" width="800">

In diffusion model, image generation is to starting from an initial vector $z$, after the model network with conditions map onto a distribution which is approximate to the real image distribution.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/diffusion_model_distribution.png" width="800">


Our goal is to find a model network $\Theta$ which gives $P_{\Theta}(x)$, approximate to $P_{data}(x)$. 

This is a maximum likelihood estimation process.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/DiT/Diffusion/images/maximum_likelihood_estimation.png" width="800">
