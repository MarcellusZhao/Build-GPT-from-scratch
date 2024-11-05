import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyper-parameters
context_len = 8
batch_size = 32
lr=1e-3
max_iters = 5000
eval_interval = 300
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32
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

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores, so called "affinities".
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -------> (B, T, T)
        # slicing is necessary because otherwise the dimension of tril matrix is possibly mismatched with the input in inference.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, value=float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) ------> (B, T, C)

        return out
    

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# implement the simplist language model: Bigram Language model
class BigramLanguageModel(nn.Module):

    def __init__(self): 
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_len, n_embd)
        # self.sa_head = Head(n_embd)
        self.sa_head = MultiHeadAttention(4, n_embd//4) # i.e., 4 heads of self-attention
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        _, T = idx.shape

        # This is not valid logits any more if the dimension of token embedding does not equal to vocab_size. So we need another linear layer.
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.sa_head(x) # apply one head of self-attention, (B,T,C)
        x = self.ffwd(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            # crop idx to the context window size.
            idx_cond = idx[:, :context_len]
            # get the prediction
            logits, _ = self(idx_cond)
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
m = BigramLanguageModel()
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
