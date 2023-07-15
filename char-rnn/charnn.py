import numpy as np
from tqdm import tqdm
import torch

# Load data
def load_data():
    dataset = open("datasets/shakespeare/input.txt").read()
    dataset = " ".join([line for line in dataset.splitlines() if ':' not in line]).lower() # Remove all the lines with speaker names, all lowercase, should increase consistency
    return dataset

# build the vocabulary of characters and mappings to/from integers
def build_vocab(dataset):
    chars = sorted(list(set(''.join(dataset))))
    stoi = {s:i for i,s in enumerate(chars)} # Convert character to integer representation
    itos = {i:s for s,i in stoi.items()} # Convert integer to character representation
    return stoi, itos

class CharRNN():
    def __init__(self, vocab_size, hidden_size, embedding_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        # Embeddings Layer
        self.e = torch.randn((vocab_size, embedding_size)) # embedding layer

        # Hidden Layer 1
        self.Wxh = torch.randn(embedding_size, hidden_size[0]) * 0.01 # input to hidden
        self.Whh = torch.randn(hidden_size[0], hidden_size[0]) * 0.01 # hidden to hidden
        self.bh = torch.zeros(1, hidden_size[0]) # hidden bias

        # Hidden Layer 2
        self.Wxh2 = torch.randn(hidden_size[0], hidden_size[1]) * 0.01 # prev to hidden
        self.Whh2 = torch.randn(hidden_size[1], hidden_size[1]) * 0.01 # hidden to hidden
        self.bh2 = torch.randn(1, hidden_size[1])

        # Output Layer
        self.Wo = torch.randn(hidden_size[1], vocab_size) * 0.01 # hidden to output
        self.bo = torch.randn(1, vocab_size) # output bias

        self.h1 = torch.zeros(1, hidden_size[0]) # hidden state 1
        self.h2 = torch.zeros(1, hidden_size[1]) # hidden state 2

    def forward(self, ix):
        x = self.e[ix]
        h1 = self.h1
        h2 = self.h2
        h1 = h1 = (x @ self.Wxh  + h1 @ self.Whh + self.bh).tanh() # compute hidden 1
        h2 = (h1 @ self.Wxh2 + h2 @ self.Whh2 + self.bh2).tanh()
        y = h2 @ self.Wo + self.bo # Compute output
        p = y.softmax(-1)
        self.h1.data = h1.data
        self.h2.data = h2.data
        return p
    
    def sample(self, seed_ix, n):
        if not isinstance(seed_ix, list):
            seed_ix = [seed_ix]
        
        h1 = torch.zeros_like(self.h1)
        h2 = torch.zeros_like(self.h2)
        y = torch.zeros(self.vocab_size, 1)
        probs = []
        ixes = seed_ix + []
        
        # Input iniital values
        for ix in seed_ix:
            x = self.e[ix]
            h1 = (x @ self.Wxh  + h1 @ self.Whh + self.bh).tanh() # compute hidden 1
            h2 = (h1 @ self.Wxh2 + h2 @ self.Whh2 + self.bh2).tanh()
            y = h2 @ self.Wo + self.bo # Compute output

        p = y.softmax(-1)
        probs.append(p)
        # Decide next char
        pdata = p.detach().numpy()
        ix = np.random.choice(range(self.vocab_size), p=pdata.ravel())
        ixes.append(ix)
        x = self.e[ix]

        for _ in range(n-1):
            h1 = (x @ self.Wxh  + h1 @ self.Whh + self.bh).tanh() # compute hidden 1
            h2 = (h1 @ self.Wxh2 + h2 @ self.Whh2 + self.bh2).tanh()
            y = h2 @ self.Wo + self.bo # Compute output
            p = y.softmax(-1)
            probs.append(p)

            # Decide next char
            pdata = p.detach().numpy()
            ix = np.random.choice(range(self.vocab_size), p=pdata.ravel())

            ixes.append(ix)
            x = self.e[ix] # Get embedding for predicted char

        return ixes

    def parameters(self):
        return [self.e, self.Wxh, self.Whh, self.Wxh2, self.Whh2, self.bh2, self.bh, self.Wo, self.bo]
    
       
if __name__ == '__main__':

    hidden_size = [300, 250] # Size of Recurrent layers 1 and 2
    embedding_size = 10 # Size of embeddings in embeddings layer
    seq_len = 25 # Sequence length during training
    lr = 0.1 # Learning rate
    lr_decay = 0.8 # lr = lr * lr_decay (Set to 1.0 for no decay)
    lr_min = 0.02 # mininmum lr
    epochs = 25 # Number of epochs
    sample_input = 'The ' # Input sequence to sample after each epoch

    dataset = load_data() # Returns string with entire input file
    stoi, itos = build_vocab(dataset) # Vocabulary
    dataset_np = np.array(list(dataset)) # List of each char in dataset
    train_set = dataset_np[:int(len(dataset_np) * 1.0)]
    # dev_set = dataset_np[int(len(dataset_np) * 0.7):int(len(dataset_np) * 0.85)]
    # test_set = dataset_np[:int(len(dataset_np) * 0.85)]
    
    model = CharRNN(len(stoi.keys()),hidden_size, embedding_size)
    print(f"parmeters: {sum([torch.numel(n) for n in model.parameters()])}")

    for p in model.parameters():
        p.requires_grad = True

    prev_loss = [1e6]
    lowered_lr = False
    ie = 0
    while True:
        ie += 1
        print(f"Epoch {ie} lr={lr}")
        model.h1 = torch.zeros_like(model.h1)
        model.h2 = torch.zeros_like(model.h2)
        loss_track = []
        for ix in tqdm(np.arange(0,len(train_set) - seq_len - 1, seq_len)):
            train_ex = np.arange(ix, ix+seq_len)
            train_ex = np.vectorize(stoi.get)(train_set[train_ex]) # ints for each char
            probs = []
            for i in train_ex:
                probs.append(model.forward(i))

            # Calculate Loss
            log_likelihood = torch.zeros(())

            ys = []
            for idx, l in enumerate(probs):
                Y = stoi[train_set[ix + idx + 1]]
                P = l[0,Y] # prob for expected label
                P = P.log() # log prob

                log_likelihood = log_likelihood + P # log likelihood


            nll = -log_likelihood.reshape(()) / seq_len

            # Zero grads
            for p in model.parameters():
                p.grad = None

            # Loss.backward
            nll.backward()
            loss_track.append(nll.data) # Track losses on each training step

            # Update Parameters
            for p in model.parameters():
                assert p.grad is not None, "Params must have gradients"
                p.data += -p.grad * lr


        epoch_loss = sum(loss_track) / len(loss_track)
        print(f"Epoch {ie } loss: {epoch_loss}")
        print(f"Sample: {''.join([itos[x] for x in model.sample([stoi[x] for x in sample_input], 100)])}")
        if epoch_loss >= prev_loss[len(prev_loss) - 1]:
            print(f"Reducing loss by {1 - lr_decay}")
            if lowered_lr: break
            lr *= lr_decay
            lowered_lr = True
        else:
            lowered_lr = False
            
        prev_loss.append(epoch_loss)


    ixes = model.sample([stoi[x] for x in sample_input], 200)
    print(f"End Sample: {''.join([itos[x] for x in model.sample([stoi[x] for x in sample_input], 500)])}")