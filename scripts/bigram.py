import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyper-parameters
context_len = 8
batch_size = 32
lr=1e-3
max_iters = 3000
eval_interval = 300
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------

# Start with the tiny shakespeare dataset to train on
with open('../data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in this text dataset.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {s:i for i,s in enumerate(chars)} # string character to index
itos = {i:s for i,s in enumerate(chars)} # index to string characters
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# convert data to torch.tensor and split train and val sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # size of train set
train_data = data[:n]
val_data = data[n:]

# define some utility functions
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - context_len, (batch_size,))
    x = torch.stack([data[i:i+context_len] for i in ix])
    y = torch.stack([data[i+1:i+context_len+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def evaluate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = m(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# implement the simplist language model: Bigram Language model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = 0
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # Bigram Language Model only attains to the last token in the context
            logits = logits[:, -1, :]
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


# training
m = BigramLanguageModel(vocab_size)
m = m.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
for step in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        losses = evaluate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of train data
    xb, yb = get_batch("train")
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the trained model
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))
