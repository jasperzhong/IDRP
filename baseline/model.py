'''
baseline
just two LSTMs
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_len, n_layers=1):
        super(Model, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.lstm_1 = nn.LSTM(
            embed_size,
            hidden_size,
            n_layers,
            bidirectional=True
        )

        self.lstm_2 = nn.LSTM(
            embed_size,
            hidden_size,
            n_layers,
            bidirectional=True
        )

        self.linear1 = nn.Linear(
            hidden_size * 2 * seq_len,
            hidden_size * 2
        )

        self.linear2 = nn.Linear(
            hidden_size * 2,
            100
        )

        self.linear3 = nn.Linear(
            100,
            4
        )

    def forward(self, arg1, arg2):
        '''
        Description:
            encode arg1 and arg2, then concat them together. Finally use linear layer to 
            classify.
        Inputs:
            arg1: [T * B]
            arg2: [T * B]
        '''
        # [T * B] -> [T * B * E]
        embeded_1 = self.embedding(arg1)
        embeded_2 = self.embedding(arg2)
        
        # [T * B * E] -> [T * B * 2H]
        outputs1, hidden1 = self.lstm_1(embeded_1, None)
        outputs2, hidden2 = self.lstm_2(embeded_2, None)

        # [T * B * 2H] -> [T * B * H]
        outputs1 = (
            outputs1[:, :, :self.hidden_size] + 
            outputs1[:, :, self.hidden_size:]
        )

        outputs2 = (
            outputs2[:, :, :self.hidden_size] + 
            outputs2[:, :, self.hidden_size:]
        )

        # [T * B * 2H] -> [B * T * 2H] -> [B * T x 2H]
        output = torch.cat([outputs1, outputs2], 2).transpose(0, 1).contiguous()
        output = output.view(output.size(0), -1)
        
        
        # [B * T x 2H] -> ... -> [B * 4] 
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        output = F.log_softmax(output, dim=1)

        return output



'''
if __name__=='__main__':
    vocab_size = 1000
    embed_size = 300
    hidden_size = 256

    model = Model(vocab_size, embed_size, hidden_size, 100)
    
    arg1 = torch.randint(0, 1000, size=(100, 32), dtype=torch.long)
    arg2 = torch.randint(0, 1000, size=(100, 32), dtype=torch.long)
    output = model(arg1, arg2)
    print(output[0])
'''