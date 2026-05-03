# Character-Level Language Model from Scratch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Overview
This project is a deep learning experiment built from scratch using PyTorch. The goal was to understand the core mechanics of Large Language Models (LLMs) by building a character-level text generator. 

By feeding the model raw text (e.g., Shakespearean plays), it learns the fundamental rules of English grammar, vocabulary, and punctuation, predicting text one character at a time.

## Key Features

* **Multiple Architectures Supported:** 
    * Feed-Forward Neural Network (Baseline)
  * Recurrent Neural Network (RNN)
  * Gated Recurrent Unit (GRU)
  * Long Short-Term Memory (LSTM)

* **Object-Oriented Design:** Built around an abstract `Mlm` base class for highly modular and reusable code.
* **Dynamic Generation:** Includes a custom text generation module featuring:
  * **Sliding Context Window:** Efficient memory management during inference.
  * **Temperature Scaling:** Adjustable parameter to control the creativity/randomness of the generated text.
  * **Multinomial Sampling:** Prevents repetitive loops using probability distributions.

## How It Works

The model is trained on a Many-to-Many architecture. It takes a sequence of characters, converts them into dense vectors using an `nn.Embedding` layer, processes them through the chosen temporal architecture (e.g., LSTM), and outputs logits for the next likely character across the entire vocabulary.

## Exemple output

The `temperature` parameter controls the randomness of the model's predictions. Lower values produce safer, more structured text, while higher values lead to more creative (and sometimes chaotic) word generation. Here is the model generating text at three different temperatures:

**Temperature = 0.7 (Safe & Structured)**
> first servingman<br>
> pray you, look to you.<br>
> 
> pompey<br>
> if the deeper than he shall stand for calm these fiery burgener'd.<br>
> 
> john of gaunt<br>
> good masters, make me hence, and a worse than seen,and beat them to your sa

**Temperature = 1.0 (Balanced)**
> first court-woman of their princely father holdor edward well, unmanner'd foot, a vermony,there is another worse and unruly barnerwill plant high heart's death.
> 
> claudio<br>
> deed, as it is, it is true by you

**Temperature = 1.5 (Word Salad)**
> first must i hope this scratches nogy that gloucester lucenties too.<br>
> 'forbats on the kindmenes pent in jace, just do,<br>
> thou hast, that brought within the guilty vice<br>
> signior baptista, lest you depart with ea

*Notice how at T=0.7, the model perfectly respects the format of character names and dialogue. As temperature increases to 1.5, it begins taking risks, inventing Shakespearean-sounding words like "vermony" or "kindmenes".*

*Note: While the grammar and vocabulary ("weary tears", "wouldst") are accurate for a character-level model, the logical coherence remains limited by the context window.*