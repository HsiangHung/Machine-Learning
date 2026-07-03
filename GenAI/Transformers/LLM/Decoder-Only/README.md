# How Transformers Work

Below we will have a step-by-step instruction how transformer works in a decoder-only language model, which is used for **ChatGPT**. 

We followed the StatQuest video: [Decoder-Only Transformers, ChatGPTs specific Transformer, Clearly Explained!!!](https://www.youtube.com/watch?v=bQ5BoolX9Ag) to explain how how a decoder-only transformer can take a simple input prompt, and generate a simple response.

The explanation also covers detailed computation step by step using more math way. We used all numbers shown in the above video. Meanwhile, the notebook under this directory has the correspodning transformer code.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/LLM/Decoder-Only/images/decoder-only_diagram.png" width="1000">


### Example:

In ChatGPT:

* A: What is StatQuest?
* B: Awesome.

Now we have a simple vocabulary and the **tokens** are: "what", "is", "StatQuest", "awesome", "< EOS >".
Here EOS stands as "end of sentence" or "end of sequence". Therefore we can simply represent the tokens as vectors:

$$
w["what"] = \begin{bmatrix} 
1  \\
0  \\
0  \\
0  \\
0
\end{bmatrix}, \ \ 
w["is"] = \begin{bmatrix} 
0  \\
1  \\
0  \\
0  \\
0
\end{bmatrix}, \ \ 
w["StatQuest"] = \begin{bmatrix} 
0  \\
0  \\
1  \\
0  \\
0
\end{bmatrix}, \cdots
$$

In a decoder-only transformer, we need to process the following components:
1. Word Embedding
2. Positional Encoding
3. Masked Self-Attention
4. Residual Connection
5. Fully Connect Layer + SoftMax

## 1. Word Embedding

Suppose we already have a simple D=2 word embedding (assume the embedding model was trained before) which can convert the input token to numbers:

$$
E = \begin{bmatrix} 
-2.38 & 0.61 & -2.38 & 2.21 & 2.92 \\
0.1 & 0.17 & 0.1 & -0.64 & -2.97 
\end{bmatrix}.
$$

and the activation function is just a linear function, e.g. $f(x)=x$. A word embedding is givn by $f(E \cdot w(". ."))$. 

Thus the word embeddings of the above tokens are


$$
e("what") =  \mathbf{E} w["what"] = \begin{bmatrix} 
-2.38 & 0.61 & -2.38 & 2.21 & 2.92 \\
0.1 & 0.17 & 0.1 & -0.64 & -2.97 
\end{bmatrix} \begin{bmatrix} 
1  \\
0  \\
0  \\
0  \\
0
\end{bmatrix} = \begin{bmatrix} 
-2.38  \\
0.1
\end{bmatrix}.
$$

$$
e("is") = E w["is"] = \begin{bmatrix} 
0.61  \\
0.17
\end{bmatrix},
$$

$$
e("StatQuest") = E w["StatQuest"] = \begin{bmatrix} 
-2.38  \\
0.1
\end{bmatrix}.
$$

## 2. Positional Encoding

The positional encoding functions are like

![positional_encoding_squiggle](images/positional_encoding_squiggle.png)

Since embedding dimension D = 2, we only need to look up first two plots, and first three token positions. Thus the inputs of the tokens into the transformer are

$$
"what" = \begin{bmatrix} 
1  \\
0  \\
0  \\
0  \\
0  
\end{bmatrix}
\to 
\begin{bmatrix} 
-2.38  \\
0.1
\end{bmatrix} +  
\begin{bmatrix} 
0  \\
1
\end{bmatrix} = 
\begin{bmatrix} 
-2.38  \\
1.1
\end{bmatrix},
$$

$$
"is" = \begin{bmatrix} 
0  \\
1  \\
0  \\
0  \\
0  
\end{bmatrix}
\to 
\begin{bmatrix} 
0.61  \\
0.17
\end{bmatrix} +  
\begin{bmatrix} 
0.84  \\
0.54
\end{bmatrix} = 
\begin{bmatrix} 
1.45  \\
0.71
\end{bmatrix},
$$

$$
"statQuest" = \begin{bmatrix} 
0  \\
0  \\
1  \\
0  \\
0  
\end{bmatrix}
\to 
\begin{bmatrix} 
-2.38  \\
0.1
\end{bmatrix} +  
\begin{bmatrix} 
0.9  \\
-0.42
\end{bmatrix} = 
\begin{bmatrix} 
-1.47  \\
-0.32
\end{bmatrix},
$$


## 3. Masked Self-Attention

Then we need to masked self-attention

$$ \textrm{SoftMax} \Big( \frac{Q[w_1] K[w_2]^T}{\sqrt{d}} \Big) V[w_2] = Pr(w_1, w_2) V[w_2], $$

where $\sqrt{d}$ is the normalization for embedding dimension, thus $\sqrt{d}=\sqrt{2}$. However, in the following, for simplicity, I just use $\sqrt{d}=1$. (This doesn't impact results of attentions.)

For **decoder-only** transformer, we only need masked self-attentions, only the tokens prior to the query. 

For example, to compute the masked self-attention of "is", we only need to consider the tokens "The pizza .... and".

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/LLM/Decoder-Only/images/masked_self_attention.png" width="1000">


For encode transformer we still need self-attention.

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/LLM/Decoder-Only/images/self_attention.png" width="1000">

### 3.1 Q, K, V

Assume we have Q/K/V matrices (either from trained or randomly initialized) as 

$$
Q = \begin{bmatrix} 
-0.8 & -1.7 \\
0.4 & 2.8  
\end{bmatrix}, \ \ \ 
K = \begin{bmatrix} 
-1.5 & 0.7 \\
1.5 & -2.1  
\end{bmatrix}, \ \ \ 
V = \begin{bmatrix} 
1 & -0.5 \\
0.6 & 0.1  
\end{bmatrix}.
$$ 



**Query** for "what", "is", and "StatQuest" are

$$
Q["what"] = \begin{bmatrix} 
-0.8 & -1.7 \\
0.4 & 2.8  
\end{bmatrix}
\begin{bmatrix} 
-2.38 \\
1.1
\end{bmatrix}
\sim \begin{bmatrix} 
0 \\
2.1
\end{bmatrix}.
$$

$$
Q["is"] = \begin{bmatrix} 
-0.8 & -1.7 \\
0.4 & 2.8  
\end{bmatrix}
\begin{bmatrix} 
1.45 \\
0.71
\end{bmatrix}
\sim \begin{bmatrix} 
-2.4 \\
2.6
\end{bmatrix}.
$$

$$
Q["StatQuest"] = \begin{bmatrix} 
-0.8 & -1.7 \\
0.4 & 2.8  
\end{bmatrix}
\begin{bmatrix} 
-1.47 \\
-0.32
\end{bmatrix}
\sim \begin{bmatrix} 
1.7 \\
-1.5
\end{bmatrix}.
$$


**Key** for "what", "is" and "StatQuest" are

$$ 
K["what"] = \begin{bmatrix} 
-1.5 & 0.7 \\
1.5 & -2.1  
\end{bmatrix}
\begin{bmatrix} 
-2.38 \\
1.1
\end{bmatrix}
\sim \begin{bmatrix} 
4.3 \\
-5.9
\end{bmatrix},
$$
$$
K["is"] = \begin{bmatrix} 
-1.5 & 0.7 \\
1.5 & -2.1  
\end{bmatrix}
\begin{bmatrix} 
1.45 \\
0.71
\end{bmatrix}
\sim \begin{bmatrix} 
-1.7 \\
0.7
\end{bmatrix}.
$$
$$
K["StatQuest"] = \begin{bmatrix} 
-1.5 & 0.7 \\
1.5 & -2.1  
\end{bmatrix}
\begin{bmatrix} 
-1.47 \\
-0.32
\end{bmatrix}
\sim \begin{bmatrix} 
2 \\
-1.5
\end{bmatrix}.
$$


**Value** for "what", "is" and "StatQuest" are

$$ 
V["what"] = \begin{bmatrix} 
1 & -0.5 \\
0.6 & 0.1  
\end{bmatrix}
\begin{bmatrix} 
-2.38 \\
1.1
\end{bmatrix}
\sim \begin{bmatrix} 
-2.9 \\
-1.3
\end{bmatrix},
$$
$$
V["is"] = \begin{bmatrix} 
1 & -0.5 \\
0.6 & 0.1  
\end{bmatrix}
\begin{bmatrix} 
1.45 \\
0.71
\end{bmatrix}
\sim \begin{bmatrix} 
1.1 \\
0.9
\end{bmatrix}.
$$
$$
V["StatQuest"] = \begin{bmatrix} 
1 & -0.5 \\
0.6 & 0.1  
\end{bmatrix}
\begin{bmatrix} 
-1.47 \\
-0.32
\end{bmatrix}
\sim \begin{bmatrix} 
-1.3 \\
-0.9
\end{bmatrix}.
$$


### 3.2 Similarity

The similarity between $Q["is"]$ and K for "what" is

$$ \langle Q["is"], K["what"] \rangle = \begin{bmatrix} 
-2.4 & 2.6  
\end{bmatrix} 
\begin{bmatrix} 
4.3 \\
-5.9  
\end{bmatrix} \sim -25.$$

K for "is" is

$$ \langle Q["is"], K["is"] \rangle = \begin{bmatrix} 
-2.4 & 2.6  
\end{bmatrix} 
\begin{bmatrix} 
-1.7 \\
0.7  
\end{bmatrix} = 5.9.$$

### 3.3 SoftMax

The softMax probabilities are

$$ Pr("is", "what") = \textrm{SoftMax} \Big( Q["is"] K["what"]^T \Big) = \frac{e^{-25}}{e^{5.9} + e^{-25}} \sim 0. $$
$$ Pr("is", "is") = \textrm{SoftMax} \Big( Q["is"] K["is"]^T \Big) = \frac{e^{5.9}}{e^{5.9} + e^{-25}} \sim 1. $$


### 3.4 Masked Self-Attention

The masked self-attention for "is" is

$$ \textrm{SoftMax} \Big( \frac{Q K^T}{\sqrt{d}} \Big) V =  Pr("is", "what")V["what"] + Pr("is", "is") V["is"] = \begin{bmatrix} 
-2.9 \\
-1.3
\end{bmatrix} \times 0 + 
\begin{bmatrix} 
1.1 \\
0.9
\end{bmatrix} \times 1 = \begin{bmatrix} 
1.1 \\
0.9
\end{bmatrix}.
$$

For "what", only "what" is prior to it. Thus

$$ \textrm{SoftMax} \Big( \frac{Q K^T}{\sqrt{d}} \Big) V =  Pr("what", "what")V["what"]  = \begin{bmatrix} 
-2.9 \\
-1.3
\end{bmatrix},
$$

where $Pr("what", "what")=1$.

For "StatQuest"

$$ 
\begin{aligned}
\textrm{SoftMax} \Big( \frac{Q K^T}{\sqrt{d}} \Big) V &=  Pr("StateQuest", "what") \begin{bmatrix} 
-2.9 \\
-1.3
\end{bmatrix} + Pr("StateQuest", "is") \begin{bmatrix} 
1.1 \\
0.9
\end{bmatrix} + Pr("StateQuest", "StateQuest") \begin{bmatrix} 
-1.3 \\
-0.9
\end{bmatrix} \\ 
 &\sim 1 \times \begin{bmatrix} 
-2.9 \\
-1.3
\end{bmatrix} + 0 \times \begin{bmatrix} 
1.1 \\
0.9
\end{bmatrix} + 0 \times \begin{bmatrix} 
-1.3 \\
-0.9
\end{bmatrix} = \begin{bmatrix} 
-2.9 \\
-1.3
\end{bmatrix}.
\end{aligned}
$$

## 4. Residual Connection

Sum over word embeddings, positional encodings and masked self-attentions, we have residual connections

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/LLM/Decoder-Only/images/resid_connection.png" width="1000">


## 5. Fully Connect Layer + SoftMax

Given residual connects for the tokens, assume we have trained fully connected layer (MLP), we can compute the probability of each token.

In the below example, we use the weights of the fully connect layer as

$$
\begin{bmatrix} 
1.5 & 1.2 \\
-2.5 & 4.1 \\
1.2 & 1.6 \\
1.7 & -3.0 \\
-0.6 & -1.5 \\
\end{bmatrix}.
$$

Note there are 5 tokens in our vocabulary, thus the layer connects D=2 to 5 tokens.

For example, for input token "what" and "is":

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/LLM/Decoder-Only/images/fully_connect_layer-1.png" width="800">


We see the first predicted token is correct, but the second slightly deviates from our expectation.

For token "StatQuest":

<img src="https://github.com/HsiangHung/Machine-Learning/blob/master/GenAI/Transformers/LLM/Decoder-Only/images/fully_connect_layer-2.png" width="350">

We see the model predicts it is the end of sentence.