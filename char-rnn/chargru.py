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


class GRUCell():
    def __init__(self, input_size, hidden_size, device='cpu') -> None:

        # Candidate State
        self.Wc = torch.randn((hidden_size, hidden_size + input_size)) * 0.01 
        self.bc = torch.zeros((hidden_size, 1), device=device)

        # Update Gate
        self.Wu = torch.randn((hidden_size, hidden_size + input_size)) * 0.01
        self.bu = torch.zeros((hidden_size, 1), device=device)

        # Reset Gate
        self.Wr = torch.randn((hidden_size, hidden_size + input_size)) * 0.01
        self.br = torch.zeros((hidden_size, 1), device=device)

    def forward(self, x, cprev):
        gu = (self.Wu @ torch.cat((cprev, x)) + self.bu).sigmoid() # Gamma u ( Update Gate )
        gr = (self.Wr @ torch.cat((cprev, x)) + self.br).sigmoid() # Gamma r ( Reset Gate )
        c_cand = (self.Wc @ torch.cat((gr * cprev, x)) + self.bc).tanh() # c~, ( Candidate State )
        c_new = gu * c_cand + (1-gu) * cprev # New State

        return c_new
    
    def parameters(self):
        return [self.Wc, self.bc, self.Wu, self.bu, self.Wr, self.br]

class CharRNNGRU():
    def __init__(self, vocab_size, hidden_size, embedding_size, device='cpu'):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.device = device

        # Embeddings Layer
        self.e = torch.randn((vocab_size, embedding_size), device=device)

        # Hidden Layer
        self.gru = GRUCell(embedding_size, hidden_size[0], device=device)
        self.gru2 = GRUCell(hidden_size[0], hidden_size[1], device=device)

        # FF Layer
        self.Wo = torch.randn((vocab_size, hidden_size[1]), device=device)
        self.bo = torch.zeros((vocab_size, 1), device=device)

        self.c = torch.zeros((hidden_size[0], 1), device=device) # Hidden state vector
        self.c2 = torch.zeros((hidden_size[1], 1), device=device) 

    def forward(self, ix):
        c = self.c
        c2 = self.c2

        c_next = self.gru.forward(self.e[ix].contiguous().view(-1,1), c)
        c2_next = self.gru2.forward(c_next, c2)
        self.c.data = c_next.data
        self.c2.data = c2_next.data
        
        y = self.Wo @ c2_next + self.bo

        p = y.softmax(0)
        return p
    
    def parameters(self):
        return [self.Wo, self.bo, self.e] + self.gru.parameters()
    
    def sample(self, seed_ix, n):
        if not isinstance(seed_ix, list):
            seed_ix = [seed_ix]
        
        c = torch.zeros_like(self.c, device=self.device)
        c2 = torch.zeros_like(self.c2, device=self.device)
        y = torch.zeros(self.vocab_size, 1, device=self.device)
        probs = []
        ixes = seed_ix + []
        
        # Input iniital values
        for ix in seed_ix:
            x = self.e[ix]
            c = self.gru.forward(self.e[ix].contiguous().view(-1,1), c)
            c2 = self.gru2.forward(c, c2)

            y = self.Wo @ c2 + self.bo


        p = y.softmax(0)
        probs.append(p)
        # Decide next char
        pdata = p.detach().numpy()
        ix = np.random.choice(range(self.vocab_size), p=pdata.ravel())
        ixes.append(ix)
        x = self.e[ix].contiguous().view(-1,1)

        for _ in range(n-1):
            c = self.gru.forward(x, c)
            c2 = self.gru2.forward(c, c2)

            y = self.Wo @ c2 + self.bo
            p = y.softmax(0)
            probs.append(p)

            # Decide next char
            pdata = p.detach().numpy()
            ix = np.random.choice(range(self.vocab_size), p=pdata.ravel())

            ixes.append(ix)
            x = self.e[ix].contiguous().view(-1,1) # Get embedding for predicted char

        return ixes

    
       
if __name__ == '__main__':

    hidden_size = [250, 250] # Size of Recurrent layers 1 and 2
    embedding_size = 10 # Size of embeddings in embeddings layer
    seq_len = 25 # Sequence length during training
    lr = 0.1 # Learning rate
    lr_decay = 0.8 # lr = lr * lr_decay (Set to 1.0 for no decay)
    lr_min = 0.02 # mininmum lr
    epochs = 25 # Number of epochs
    sample_input = 'the ' # Input sequence to sample after each epoch

    dataset = load_data() # Returns string with entire input file
    stoi, itos = build_vocab(dataset) # Vocabulary
    dataset_np = np.array(list(dataset)) # List of each char in dataset
    train_set = dataset_np[:int(len(dataset_np) * 1.0)]
    # dev_set = dataset_np[int(len(dataset_np) * 0.7):int(len(dataset_np) * 0.85)]
    # test_set = dataset_np[:int(len(dataset_np) * 0.85)]
    
    model = CharRNNGRU(len(stoi.keys()),hidden_size, embedding_size)
    print(f"parmeters: {sum([torch.numel(n) for n in model.parameters()])}")

    for p in model.parameters():
        p.requires_grad = True

    prev_loss = [1e6]
    lowered_lr = False
    ie = 0
    while True:
        ie += 1
        print(f"Epoch {ie} lr={lr}")
        model.c = torch.zeros_like(model.c)
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
                P = l[Y,0] # prob for expected label
                P = P.log() # log prob

                log_likelihood = log_likelihood + P # log likelihood


            nll = -log_likelihood.reshape(()) / seq_len

            # Zero grads
            for p in model.parameters():
                p.grad = None

            loss_track.append(nll.data) # Track losses on each training step

            # Loss.backward
            nll.backward()

            # Update Parameters
            for p in model.parameters():
                assert p.grad is not None, "Params must have gradients"
                p.data += -p.grad * lr


        epoch_loss = sum(loss_track) / len(loss_track)
        print(f"Epoch {ie } loss: {epoch_loss}")
        print(f"Sample: {''.join([itos[x] for x in model.sample([stoi[x] for x in sample_input], 250)])}")
        if epoch_loss >= prev_loss[len(prev_loss) - 1]:
            print(f"Reducing loss by {1 - lr_decay}")
            if lowered_lr: break
            lr *= lr_decay
            lowered_lr = True
        else:
            lowered_lr = False
            
        prev_loss.append(epoch_loss)

    print(f"End Sample: {''.join([itos[x] for x in model.sample([stoi[x] for x in sample_input], 500)])}")