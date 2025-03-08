import torch
import torch.nn as nn # neural network. This import is necessary for defining the model
from torch.nn import functional as F # This import is necessary for using activation functions

#hyperparameters
# -------------------------------
max_iters = 3000 # This is the maximum number of iterations that the model is going to train for
eval_interval = 300 # This is the interval at which the model is going to evaluate the loss
batch_size = 32 # This is the number of sequences that the model is going to train on at a time
block_size = 8 # This is the maximum sequence length that the model can train on at a time
learning_rate = 1e-2 # This is the learning rate of the model
device = 'cuda' if torch.cuda.is_available() else 'cpu' # This is going to check if a GPU is available and set the device to 'cuda' if it is and 'cpu' if it is not
eval_iters = 200 # This is the number of iterations that the model is going to evaluate the loss for
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


# lets now feed this into a neural network

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # This is going to create an embedding table that is going to be used to store the embeddings of the tokens
        # the embedding table is a matrix of size vocab_size by vocab_size where each row is the embedding of a token
    def forward(self, idx, targets=None):
        #idx and the targets are btoh (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # Get the logits from the token embedding table (Batch, Time, Channel) tensor
        
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
            # get the predictions
            logits, loss = self(idx)
            
            # focus only on the last time step. We do this because the -1 is the last element in the time dimension which represents what comes next
            logits = logits[:, -1, :] # becomes (B, C)
            
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1). What this does is that it samples from the distribution of the probabilities. This represents a single prediction of what comes next
            
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1). What this does is that it concatenates the index with the next index
        return idx

model = BigramLanguageModel(vocab_size) # Create an instance of the BigramLanguageModel and store it in the variable m
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
    
    if iter % eval_interval == 0:
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