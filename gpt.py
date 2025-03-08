import torch
import torch.nn as nn # neural network. This import is necessary for defining the model
from torch.nn import functional as F # This import is necessary for using activation functions

#hyperparameters
# -------------------------------
max_iters = 5000 # This is the maximum number of iterations that the model is going to train for
eval_interval = 500 # This is the interval at which the model is going to evaluate the loss
batch_size = 64 # This is the number of sequences that the model is going to train on at a time
block_size = 256 # This is the maximum sequence length that the model can train on at a time
learning_rate = 3e-4 # This is the learning rate of the model
device = 'cuda' if torch.cuda.is_available() else 'cpu' # This is going to check if a GPU is available and set the device to 'cuda' if it is and 'cpu' if it is not
eval_iters = 200 # This is the number of iterations that the model is going to evaluate the loss for
n_embd = 384 # This is the number of embeddings that the model is going to use
n_head = 6 # This is the number of heads that the model is going to use
n_layer = 6 # This is the number of layers that the model is going to use
dropout = 0.1 # This is the dropout rate of the model
# -------------------------------

torch.manual_seed(1337) # Set the seed for reproducibility

with open('input.txt', 'r', encoding='utf-8') as f: # Open the input file in read mode (r is read mode) with proper encoding
    text = f.read() # Read the file and store it in the variable text
    
#getting the unique characters that occur in the text
chars = sorted(list(set(text))) # Get the unique characters in the text and store them in the variable chars
vocab_size = len(chars) # Get teh length of the unique characters and store it in the variable vocab_size

#developing a strategy to tokenize the characters
stoi = {ch: i for i, ch in enumerate(chars)} # Create a dictionary with the characters as keys and their index as values
itos = {i:ch for i, ch in enumerate(chars)} # Create a dictionary with the index as keys and the characters as values
encode = lambda s: [stoi[c] for c in s] # Create a lambda function that takes a string named s and returns a list of indices
decode = lambda l: ''.join([itos[i] for i in l]) # Create a lambda function that takes a list of indices and returns the characters in a string named l

#Example:
#print(encode("hii there"))
#[46, 47, 47, 1, 58, 46, 43, 46, 43]
#print(decode(encode("hii there")))
#'hii there'

# Next we got to train and do test splits
data = torch.tensor(encode(text), dtype=torch.long) # Convert the text to a tensor of indices and store it in the variable data
# now the entire dataset of text is represented as a tensor of indices (a very large sequence of integers)

# split up the data into tran and validation sets
n = int(0.9*len(data)) # this is going to grab the first 90% of the data to be the training set
train_data = data[:n] # Get the first 90% of the data and store it in the variable train_data
val_data = data[n:] # Get the last 10% of the data and store it in the variable val

# What this is going to do is help create a model that takes in 90% of the data and uses it to learn patterns to then predict the last 10% of the data

# this function is going to generate a small batch of data of inputs x and targets y
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == 'train': # If the split is train
        data = train_data # Set the data to the training data
    else: # Otherwise
        data = val_data # Set the data to the validation data
    
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Get a random integer between 0 and the length of the data minus the block size. This makes sure that the model doesn't try to predict something that is out of bounds
    x = torch.stack([data[i:i+block_size] for i in ix]) # Get a random sequence of characters of length block_size and store it in the variable x. First block size characters starting at i
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Get the next sequence of characters of length block_size and store it in the variable y. offset of 1 from x
    x,y = x.to(device), y.to(device) # Set the device of x and y to the device
    return x, y

# we train up from 1 to the length of the block size

# xb, yb = get_batch('train') # Get a batch of training data and store it in the variables xb and yb
# # xb is the input and yb is the target

# for b in range(batch_size): #batch dimension
#     for t in range(block_size): #time dimension
#         context = xb[b,:t+1] # Get the context of the input
#         target = yb[b,t] # Get the target of the input
#         print(f"when input is {context.tolist()} the target is {target}") # Print the context and the target


#Function to estimate Loss
@torch.no_grad() # This is a decorator that disables the gradient computation. This is useful when you are not training the model
def estimate_loss():
    out = {}
    model.eval() # Set the model to evaluation mode
    for split in ['train', 'val']: # For each split in the list ['train', 'val']
        losses = torch.zeros(eval_iters) # Create a tensor of zeros of size eval_iters and store it in the variable losses
        for k in range(eval_iters):
            X, Y = get_batch(split) # Get a batch of data and store it in the variables X and Y
            logts, loss = model(X, Y) # Get the output of the model and store it in the variables logts and loss
            losses[k] = loss.item() # Set the kth element of losses to the loss
        out[split] = losses.mean() # get the average loss for the split
    model.train() # Set the model to training mode
    return out

class Head(nn.Module):
    #one head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, 16)
        q = self.query(x) # (B, T, 16)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=1) # (B, T, T)
        wei = self.dropout(wei) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) #(B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
    
class MultiHeadAttention(nn.Module):
    # multiple heads of self-attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    # a simple linear layer followed by a non-linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # transformer block: communication followed by computation
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size) # communication
        self.ffwd = FeedForward(n_embd) # computation
        self.ln1 = nn.LayerNorm(n_embd) # layer norm 1
        self.ln2 = nn.LayerNorm(n_embd) # layer norm 2
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # communication
        x = x + self.ffwd(self.ln2(x)) # computation
        return x

# lets now feed this into a neural network
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # This is going to create an embedding table that is going to be used to store the embeddings of the tokens
        # the embedding table is a matrix of size vocab_size by vocab_size where each row is the embedding of a token
        
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # This is going to create an embedding table that is going to be used to store the embeddings of the positions
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # This is going to create a sequence of transformer blocks that is going to be used to perform the transformer operations
        self.ln_f = nn.LayerNorm(n_embd) # This is going to create a layer norm layer that is going to be used to normalize the output of the model
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) # This is going to create a multi-head attention layer that is going to be used to perform self-attention
        # self.ffwd = FeedForward(n_embd) # This is going to create a feedforward layer that is going to be used to perform feedforward operations
        self.lm_head = nn.Linear(n_embd, vocab_size) # This is going to create a linear layer that is going to be used to predict the next token
        
        self.apply(self.init_weights) # This is going to initialize the weights of the model
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape
        
        #idx and the targets are both (B, T) tensor of integers
        token_emb = self.token_embedding_table(idx) # Get the logits from the token embedding table (Batch, Time, Channel) tensor
        pos_emb = self.position_embedding_table(torch.arange(T, device= device)) # (T, C). Get the logits from the position embedding table
        x = token_emb + pos_emb # (B, T, C). Add the token embedding and the position embedding
        x = self.blocks(x) # (B, T, C). Perform the transformer block
        x = self.ln_f(x) # (B, T, C). Perform layer normalization
        # x = self.sa_heads(x) # (B, T, C). Perform self-attention
        # x = self.ffwd(x) # (B, T, C). Perform feedforward operations
        logits = self.lm_head(x) # Batch, Time, Vocab_size
        
        if targets is None: # If there are no targets
            loss = None # Set the loss to None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # Calculate the loss using the cross entropy loss function
            # we have the identity of the next character so how well are we predicting the next character based on the logits
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indicies in the current context
        # idx is teh current context of some characters in a batch
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, T)
            
            # get the predictions
            logits, loss = self(idx_cond) # (B, T, C)
            
            # focus only on the last time step. We do this because the -1 is the last element in the time dimension which represents what comes next
            logits = logits[:, -1, :] # becomes (B, C)
            
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1). What this does is that it samples from the distribution of the probabilities. This represents a single prediction of what comes next
            
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1). What this does is that it concatenates the index with the next index
        return idx

model = BigramLanguageModel() # Create an instance of the BigramLanguageModel and store it in the variable m
m = model.to(device) # Set the device of the model to the device

#logits, loss = m(xb, yb) # Get the output of the model and store it in the variable out
#print(logits.shape) # Print the shape of the output
#print(loss)

 # Create a tensor of zeros of size 1x1 and store it in the variable idx
#print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long),max_new_tokens=100)[0].tolist())) # Generate a sequence of characters and print it

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) # Create an instance of the Adam optimizer and store it in the variable optimizer

#training loop
for iter in range(max_iters):
    
    if iter % eval_interval == 0 or iter ==max_iters - 1:
        losses = estimate_loss()
        print(f"iter {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}") 
    
    # sample a batch of data
    xb, yb = get_batch('train') # Get a batch of training data and store it in the variables xb and yb
    
    # evaluate the loss
    logits, loss = m(xb, yb) 
    optimizer.zero_grad(set_to_none=True) # Zero the gradients of the optimizer
    loss.backward() # Backpropagate the loss
    optimizer.step() # Update the weights of the model
    
    #print(loss.item()) # Print the loss
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Create a tensor of zeros of size 1x1 and store it in the variable context    
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # Generate a sequence of characters and print it

#next we need the tokens to start talking to each other to build a better context

# torch.manual_seed(1337)
# B, T, C = 4, 8, 2
# x = torch.randn(B, T, C)
# x.shape

# #version 1
# xbow = torch.zeros((B,T,C))
# for b in range(B):
#     for t in range(T):
#         xprev = xbow[b, :t+1] # (t, C)
#         xbow[b, t] = xprev.mean(xprev, 0) 
        
# #version 2
# wei = torch.tril(torch.ones(T, T)) # weighted sum of all previous tokens
# wei = wei / wei.sum(1, keepdim=True)
# xbow2= wei @ x # (B, T, T) @ (B , T, C) -> (B, T, C)
# torch.allclose(xbow, xbow2)

# #version 3
# tril = torch.tril(torch.ones(T, T)) # lower triangular matrix
# wei = torch.zeros(T, T) # weighted sum of all previous tokens
# wei = wei.masked_fill(tril == 0, float('-inf')) # mask out the upper triangular part
# wei = F.softmax(wei, dim=1) # normalize the weights
# xbow3 = wei @ x
# torch.allclose(xbow, xbow3)

# # version 4
# torch.manual_seed(1337)
# B, T, C = 4, 8, 32
# x = torch.randn(B, T, C)

# #lets see a single head perform self-attention
# head_size = 16
# key = nn.Linear(C, head_size, bias=False)
# query = nn.Linear(C, head_size, bias=False)
# value = nn.Linear(C, head_size, bias=False)

# k = key(x) # (B, T, 16)
# q = query(x) # (B, T, 16)
# wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) -> (B, T, T) 

# tril = torch.tril(torch.ones(T, T)) # lower triangular matrix
# #wei = torch.zeros(T, T) # weighted sum of all previous tokens
# wei = wei.masked_fill(tril == 0, float('-inf')) # mask out the upper triangular part
# wei = F.softmax(wei, dim=1) # normalize the weights
# v = value(x) # (B, T, 16)
# out = wei @ v # (B, T, T) @ (B, T, 16) -> (B, T, 16)
# out.shape

