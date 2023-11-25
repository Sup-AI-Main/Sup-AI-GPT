#import torch and setting up cuda device

#To Activate

#cuda\Scripts\activate
#python data-extract.py


import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
import time

parser = argparse.ArgumentParser(description='Welcome to Sup.AI!')

#add arguments to the parser, with specified expected types
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a -batch_size')
parser.add_argument('-max', type=str, required=True, help='Please provide a -max(max_iters)')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#change depending on ram
#batch_size = 64
batch_size = int(args.batch_size)
block_size = 128


#max_iters = 1000
max_iters = int(args.max)
eval_interval = 500
#learningRateCombinations - 3e-3, 3e-4, 1e-3, 1e-4
learning_rate = 3e-4
eval_iters = 100
dropout = 0.2
n_embd = 384 #Vector Categories Number
n_layer = 4 #current layers dimensions for all the heads
n_head = 4 #current heads in parrellel

print(device)

#manual seed for torch

#torch.manual_seed(7777777)

#opening file to read for training
chars = ""
with open('vocab.txt', 'r', encoding='utf-8') as f:
    text =f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)

#print(vocabulary_size)

#encode and decode fuctions
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#data = torch.tensor(encode(text), dtype=torch.long)
#small_size_training

#n= int(0.9*len(data))
#train_data = data[:n]
#val_data = data[:n]
#small_size_training

#Large Scale_training, n= int(file - blocksize*batchsize, essentially 1*(0.99~99)
#Memory map for using small parts of text from a single file of any size
def get_random_chunk(split):
    filename = "Output_train.txt" if split == 'train' else "Output_val.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            #Determines file size and a random position to start
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)
            #Seek to random position and read block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)
            #Decode block to string, ignores byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r','')
            #Train and test Splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data


#get_batch
def get_batch(split):
    #(small scale)data = train_data if split == 'train' else val_data
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

x, y = get_batch('train')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #input(batch,time-setp,channels) and output(B, T, head size) size
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Adjust self.tril if necessary
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #weighted Aggregation
        v = self.value(x)
        out = wei @ v
        return out
        

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # ([Dimension=-1]Batch, Time, Channel Dimension), Makes it easier to process
        out = self.dropout(self.proj(out))
        return out

#FeedForward Class
class FeedForward(nn.Module):
    """Simple linear layer followed by non-lineraity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

#*BlockClass
class Block(nn.Module):
    """Transformer Block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        #n_embd: dimension of embedding, n_head: Number of desired heads(Models)
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    #this is the add and norm, instead of norm and add, that differs from gpt(dev)
    def forward(self, x):
        #self Attention
        y = self.sa(x)
        #add (x+y) then norm (self.ln1()), can also norm and add, (x + self.sa(self.ln1(x)))
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

#Bigram Languange model, * means changes
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) #how many decoder layers(n_layer)?
        self.ln_f = nn.LayerNorm(n_embd) #fianl layer Normalizer
        self.lm_head = nn.Linear(n_embd, vocab_size)
        #*Initializes weights using Standard Deviation, here because its used in practice
        self.apply(self.__init_weights) 

    def __init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward (self, index, targets=None):
        B, T = index.shape

        #* forwarding tokens and embedding multiple components ,index and targets are both (B,T) tensor of Integers
        tok_emb = self.token_embedding_table(index) # (B,T,C) Token Embedding
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  #(T,C) Position Embedding
        x = tok_emb + pos_emb    #-
        x = self.blocks(x)       #- (B,T,C)
        x = self.ln_f(x)         #-
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop index to the last block_size tokens:
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
print('loading')

with open('model-01.pk1', 'rb') as f:
    model = pickle.load(f)
print('loaded')

m = model.to(device)

start_time = time.time()
# Pytorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters ==0:
        losses = estimate_loss()
        print(f"step: {iter} train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate loss
    logits, loss=model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())  

with open('model-01.pk1', 'wb') as f:
    pickle.dump(model, f)
print('saved') 

end_time = time.time()

total_time = end_time - start_time

print(f'time taken: {total_time}')