"""
model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    '''
    baseline model 
    just two Bi-LSTMs
    '''
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
            hidden_size * 4 * seq_len,
            hidden_size * 4
        )

        self.linear2 = nn.Linear(
            hidden_size * 4,
            84
        )

        self.linear3 = nn.Linear(
            84,
            2
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


        # [T * B * 4H] -> [B * T * 4H] -> [B * T x 4H]
        output = torch.cat([outputs1, outputs2], 2).transpose(0, 1).contiguous()
        output = output.view(output.size(0), -1)
        
        # [B * T x 4H] -> ... -> [B * 4] 
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)

        return output
    
    def load_pretrained_embedding(self, fix_embed, glove_path, word_dict):
        embedding = self.embedding.weight.data
        if fix_embed:
            self.embedding.weight.requires_grad = False
        
        cnt = 0
        with open(glove_path, "r") as f:
            for line in f:
                parsed = line.rstrip().split(' ') 
                assert(len(parsed) == embedding.size(1) + 1)
                w = parsed[0]
                if word_dict.get(w):
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    embedding[word_dict[w]].copy_(vec)
                    cnt += 1
        print("Total embeded words %d" % cnt) 
