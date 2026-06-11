# Precision

[FP explanation in Linkedin](https://www.linkedin.com/posts/pratiush-singh-8051141a_exponent-precisiontype-fp16-share-7231290853300785152-okDk/)

[Nvidia FP8 explanation](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)

By Pratiush Singh, [FP explanation in Linkedin](https://www.linkedin.com/posts/pratiush-singh-8051141a_exponent-precisiontype-fp16-share-7231290853300785152-okDk/):


As models grow larger with the rise of LLMs, choosing the right precision type has become important.
Sharing some insights over FP16 vs BF16:
 
* FP16 and BF16 both consume same memory. 
* FP16 has better precision but lower dynamic range. In fp16, largest number is 65,504. BF16 can go upto 3.39e+38 ! 
This is because fp16 has higher # mantissa bits and bf16 has higher #exponent bits.
* For model training, you will have to experiment and understand whether your network is sensitive to range or precision. If a model requires more dynamic range, it will converge better with bf16, which has same range as fp32. However, some models will converge better with higher precision (fp16)
* When finetuning an fp16 model pre-trained in bf16, overflow issues may arise as some of the bf16 model weights might be larger than 65k.