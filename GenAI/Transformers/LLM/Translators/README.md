# Encoder + Decoder Translator


![encoder_decoder_architecture](images/translator_arch.png)

The translator code here follows the medium post: [Building a Sequence-to-Sequence Transformer Model for Language Translation](https://medium.com/@samson.sabu/building-a-sequence-to-sequence-transformer-model-for-language-translation-ac4a37533cfa)

## Understanding the Transformer Architecture

The Transformer architecture consists of an encoder-decoder structure:

* **Encoder**: The encoder processes the input sequence and converts it into a set of continuous representations, called memory, that the decoder will use to generate the output sequence.
* **Decoder**: The decoder takes the encoder’s memory and the target sequence (shifted by one token) to generate the predicted sequence token by token.

The transformer uses self-attention mechanisms to capture relationships between words in a sentence regardless of their distance, making it highly efficient for **sequence-based tasks like translation**.

**For translation task, in transformers, decoder provides queries, and encoded look up keys and values.**

In the training process, we:
* Use **Cross-Entropy Loss** to compute the loss.
* Use **Adam** as the optimizer.
* Use metrics like **BLEU score** To evaluate the model.


## The Roles in Cross-Attention

To understand why "the decoder provides the queries" and "the encoder provides the keys and values", it helps to use the analogy of a **library retrieval system**:

* The Queries ($Q$) come from the Decoder: * Think of the Query as the **search term**.The decoder looks at the words it has translated so far and asks a question: "Based on what I've written, what information do I need from the original sentence to predict the very next word?"
* The Keys ($K$) come from the Encoder: * Think of the Keys as the **tags or labels on the books in the library**.The encoder has processed the entire original source sentence. The keys represent the grammar, position, and role of every word in that original sentence (e.g., "I am a verb," or "I am the subject of the sentence").
* The Values ($V$) come from the Encoder: * Think of the Values as the actual contents of the book.Once the decoder's Query matches strongly with an encoder's Key, the Transformer pulls the corresponding Value (the rich, mathematical representation of that word's meaning) and uses it to generate the next translated word.

### A Concrete Example: French to English

Imagine translating the French sentence "Le chat noir" into English ("The black cat"):
* The Encoder processes "Le chat noir" and generates Keys and Values for all three words.
* The Decoder starts translating. Let's say it has already generated the word "The".
* The **Decoder** now generates a **Query**: "I just said 'The'. I need a noun to follow it."
* This Query is compared against all the **Keys** from the French sentence. It finds a massive mathematical match with the Key for "chat" (because "chat" is a noun).
* The model grabs the Value of "chat" (which holds the meaning "feline/cat") and pulls it into the decoder.
* The decoder outputs "cat".

(Note: In English, adjectives come first, so the model would actually match with "noir" (black) first, but the Q-K-V mechanism remains exactly the same!)

### One Quick Caveat: Self-Attention

Just keep in mind that Transformers also use Self-Attention before they do Cross-Attention
* Inside the Encoder by itself, the Encoder provides its own Queries, Keys, and Values to understand the context of the source sentence.
* Inside the Decoder by itself, the Decoder provides its own Queries, Keys, and Values to ensure the grammar of the language it is generating makes sense.

But the moment the two sides talk to each other to actually perform the translation, your logic is 100% correct: Decoder = Query, Encoder = Keys & Values.
