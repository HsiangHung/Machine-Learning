# How Transformers Work

Below we will have a step-by-step instruction how transformer works in a decoder-only language model, which is used for **ChatGPT**. 

We followed the StatQuest video: [Decoder-Only Transformers, ChatGPTs specific Transformer, Clearly Explained!!!](https://www.youtube.com/watch?v=bQ5BoolX9Ag) to explain how how a decoder-only transformer can take a simple input prompt, and generate a simple response.

The explanation also covers detailed computation step by step using more math way. We used all numbers shown in the above video. Meanwhile, the notebook under this directory has the correspodning transformer code.

### Example:

In ChatGPT:

* A: What is StatQuest?
* B: Awesome.

Now we have a simple vocabulary and the **tokens** are: "what", "is", "StatQuest", "awesome", "< EOS >".
Here EOS stands as "end of sentence" or "end of sequence". Therefore we can simply represent the tokens as vectors:

$$
w("what") = \begin{bmatrix} 
1  \\
0  \\
0  \\
0  \\
0
\end{bmatrix}, \ \ 
w("is") = \begin{bmatrix} 
0  \\
1  \\
0  \\
0  \\
0
\end{bmatrix}, \ \ 
w("StatQuest") = \begin{bmatrix} 
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
e("what") = \mathbf{A}  \mathbf{E}  w("what") = \begin{bmatrix} 
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
e("is") = E \cdot w("is") = \begin{bmatrix} 
0.61  \\
0.17
\end{bmatrix},
$$

$$
e("StatQuest") = E \cdot w("StatQuest") = \begin{bmatrix} 
-2.38  \\
0.1
\end{bmatrix}.
$$

## 2. Positional Encoding


![positional_encoding_squiggle](images/positional_encoding_squiggle.png)



