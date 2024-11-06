# Build-GPT-from-scratch

This repository contains code I wrote when I was taking the online lecture ([Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=WL&index=5)) given by Andrej Karpathy. 

## Data and tokenization

The primary dataset used for training and validation is the **tiny Shakespeare**, which contains 40,000 lines of Shakespeare from a variety of Shakespeare's plays. The tokenizer used in this mini-project is a character-level tokenizer that maps each unique character appearing in the text to an individual index number. It'd be expected that the training performance will be further improved if adopting a sub-word level tokenizer, like what OpenAI and Google do, because it makes the context of each training example more condense, lifting the learning efficiency.

## Run

Under the `scripts` folder, running `bigram.py` and `toy_gpt.py` scripts will initialize a small model that can run on CPU. `gpt.py` needs a decent GPU to run.
- `bigram.py` creates a super simple Bigram Language Model as the baseline to kick off this mini-project. It maps each character-level token to an embedding vector, and only uses the embedding vector of the last token to predict the next one, no communication involved.
- `toy_gpt.py` dramatically improves the performance over `bigram.py` by integrating multiple components, including self-attention, feed-forward layers, layer normalization, and residual connection. 
- `gpt.py` is a scaled-up version of `toy_gpt.py`, which contains around 12M trainable parameters. 

```bash
cd scripts
python bigram.py
python toy_gpt.py
python gpt.py
```

## Acknowledgements

I heartfeltly thank Andrej Karpathy for sharing this great lecture and open-sourcing the code. 

    [1] https://github.com/karpathy/ng-video-lecture