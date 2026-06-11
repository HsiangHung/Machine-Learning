# Precision


By Pratiush Singh, as models grow larger with the rise of LLMs, choosing the right precision type has become important.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Optimizaton/images/linkedin-fp-explanation.png" width="600">


From [FP explanation in Linkedin](https://www.linkedin.com/posts/pratiush-singh-8051141a_exponent-precisiontype-fp16-share-7231290853300785152-okDk/), he shared some insights over FP16 vs BF16: 
* fp16 (Half-Precision Floating Point) and bf16 (Brain Floating Point) both consume same memory; both fp16 and bf16 are exactly 16 bits (or 2 bytes) long. 
* fp16 has better precision but lower dynamic range. In fp16, largest number is 65,504. bf16 can go upto 3.39e+38.
* For model training, you will have to experiment and understand whether your network is sensitive to range or precision. If a model requires more dynamic range, it will converge better with bf16, which has same range as fp32. However, some models will converge better with higher precision (fp16)
* When finetuning an fp16 model pre-trained in bf16, overflow issues may arise as some of the bf16 model weights might be larger than 65k.

Nvidia has a blog [An Introduction to Efficient, Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) showing fp8.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Optimizaton/images/nvidia-fp-structure.png" width="800">
