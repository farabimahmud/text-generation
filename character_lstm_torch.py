import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class RNN(nn.Module):
  
  def __init__(self, input_size, output_size, hidden_size, num_layers):
    super(RNN, self).__init__()
    self.embedding = nn.Embedding(input_size, input_size)
    self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    self.decoder = nn.Linear(hidden_size,output_size)
  

  def forward(self, input_seq, hidden_state):
    embedding = self.embedding(input_seq)
    output, hidden_state = self.rnn(embedding, hidden_state)
    output = self.decoder(output)
    return output, (hidden_state[0].detach(), hidden_state[1].detach())

hidden_size = 512
seq_len = 100
num_layers = 3
lr = 0.002
epochs = 50
op_seq_len = 200
load_chk = False

save_path = "./CharRNN.pth"
data_path = "./abstract.txt"

data = open(data_path, 'r').read()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)

char_to_idx , idx_to_char = dict(), dict()
for i, ch in enumerate(chars):
  char_to_idx[ch] = i
  idx_to_char[i] = ch

data = list(data)

for i,ch in enumerate(data):
  data[i] = char_to_idx[ch]

data = torch.tensor(data).to(device)
data = torch.unsqueeze(data, dim=1)

rnn = RNN(vocab_size, vocab_size, hidden_size, num_layers).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)


for i_epoch in range(1, epochs+1):
    
    # random starting point (1st 100 chars) from data to begin
    data_ptr = np.random.randint(100)
    n = 0
    running_loss = 0
    hidden_state = None
    
    while True:
        input_seq = data[data_ptr : data_ptr+seq_len]
        target_seq = data[data_ptr+1 : data_ptr+seq_len+1]
        
        # forward pass
        output, hidden_state = rnn(input_seq, hidden_state)
        
        # compute loss
        loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
        running_loss += loss.item()
        
        # compute gradients and take optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update the data pointer
        data_ptr += seq_len
        n +=1
        
        # if at end of data : break
        if data_ptr + seq_len + 1 > data_size:
            break
        
    # print loss and save weights after every epoch
    print("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))
    torch.save(rnn.state_dict(), save_path)
    
    # sample / generate a text sequence after every epoch
    data_ptr = 0
    hidden_state = None
    
    # random character from data to begin
    rand_index = np.random.randint(data_size-1)
    input_seq = data[rand_index : rand_index+1]
    # input_seq = []
    # sentence = get_input_sentence()
    # for i in sentence:
    #   input_seq.append(data[char_to_idx[i]])
    # input_seq = torch.tensor(input_seq).to(device)

    print("----------------------------------------")
    while True:
        # forward pass
        output, hidden_state = rnn(input_seq, hidden_state)
        
        # construct categorical distribution and sample a character
        output = F.softmax(torch.squeeze(output), dim=0)
        dist = Categorical(output)
        # dist = nn.NLLLoss(output)
        index = dist.sample()
        
        # print the sampled character
        print(idx_to_char[index.item()], end='')
        
        # next input is current output
        input_seq[0][0] = index.item()
        data_ptr += 1
        
        if data_ptr > op_seq_len:
            break
        
    print("\n----------------------------------------")


# import torch
# from torchviz import make_dot
# x = torch.tensor([[1]])
# rnn = RNN(512, 512, 64, 1)
# y, hidden_state = rnn(x, None)

# make_dot(y, show_attrs=True, params=dict(rnn.named_parameters()))
# make_dot(y, show_attrs=True, params=dict(rnn.named_parameters())).render(format="png")