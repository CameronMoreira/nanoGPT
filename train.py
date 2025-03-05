import torch
import torch.nn as nn # neural network. This import is necessary for defining the model
from torch.nn import functional as F # This import is necessary for using activation functions

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

torch.manual_seed(1337) # Set the seed for reproducibility
block_size = 8 # This reprenents the maximum sequence length that the model can train on at a time
batch_size = 4 # This represents the number of sequences that the model will train on at a time

# train_data[:block_size+1] # This is going to grab the first 9 characters of the training data

# this function is going to generate a small batch of data of inputs x and targets y
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == 'train': # If the split is train
        data = train_data # Set the data to the training data
    else: # Otherwise
        data = val_data # Set the data to the validation data
    
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Get a random integer between 0 and the length of the data minus the block size. This makes sure that the model doesn't try to predict something that is out of bounds
    x = torch.stack([data[i:i+block_size] for i in ix]) # Get a random sequence of characters of length block_size and store it in the variable x
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Get the next sequence of characters of length block_size and store it in the variable y
    return x, y


