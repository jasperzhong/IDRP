import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import config 

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_len, 
                 d_a, r, n_layers=1):
        super(Model, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.r = r

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

        self.intra_attn_1 = SelfAttn(
            hidden_size, 
            d_a,
            r
        )

        self.intra_attn_2 = SelfAttn(
            hidden_size, 
            d_a,
            r
        )

        self.inter_attn = BilinearAttn(
            hidden_size
        )

        self.linear1 = nn.Linear(
            r * r,
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
        output1, _ = self.lstm_1(embeded_1, None)
        output2, _ = self.lstm_2(embeded_2, None)

        # sentence embedding
        # [T * B * 2H] -> [B x r x 2H]
        A = self.intra_attn_1(output1)
        B = self.intra_attn_2(output2)

        # attention based interaction
        # [B x r x 2H] -> [B x r x r]
        S = self.inter_attn(A, B)
        
        # flatten and mlp
        # [B x r x r] -> [B x r*r] -> [B * 4] 
        output = S.view(-1, self.r * self.r)
        output = self.linear1(output)
        output = F.log_softmax(output, dim=1)

        return output
    
    def load_pretrained_embedding(self, word_dict):
        embedding = self.embedding.weight.data
        self.embedding.weight.requires_grad = False
        cnt = 0
        with open(config.resourses.glove_path, "r") as f:
            for line in f:
                parsed = line.rstrip().split(' ') 
                assert(len(parsed) == embedding.size(1) + 1)
                w = parsed[0]
                if word_dict.get(w):
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    embedding[word_dict[w]].copy_(vec)
                    cnt += 1
        print("Total embeded words %d" % cnt) 



class SelfAttn(nn.Module):
    def __init__(self, hidden_size, d_a, r):
        super(SelfAttn, self).__init__()
        self.linear1 = nn.Linear(2 * hidden_size, d_a)
        self.linear2 = nn.Linear(d_a, r, bias=False)
        
    def forward(self, x):
        '''
        Inputs:
            x: {h_1 h_2 ... h_t}  [T x B x 2H]
        Outpus:
            M: [B x r x 2H]
        '''
        # [T x B x 2H] -> [T x B x d_a]
        w = torch.tanh(self.linear1(x))

        # [T x B x d_a] -> [T x B x r]
        w = self.linear2(w)

        # [T x B x r] -> [B x T x r] -> [B x r x T]
        w = F.softmax(w.transpose(0, 1), dim=2).transpose(1, 2)

        # [B x r x T] * [B x T x 2H] -> [B x r x 2H]
        return torch.bmm(w, x.transpose(0, 1))



class BilinearAttn(nn.Module):
    def __init__(self, hidden_size):
        super(BilinearAttn, self).__init__()
        self.linear = nn.Linear(2 * hidden_size, 2 * hidden_size)
    
    def forward(self, x, y):
        '''
        Inputs:
            x: [B x r x 2H]
            y: [B x r x 2H]
        Outputs:
            S: [B x r x r]
        '''
        # [B x r x 2H] -> [B x r x 2H] -> [B x 2H x r]
        My = self.linear(y).transpose(1, 2)

        # [B x r x 2H] * [B x 2H x r] -> [B x r x r]
        xMy = torch.bmm(x , My)

        return xMy



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