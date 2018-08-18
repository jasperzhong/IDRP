
import json
import random
import string

import torch


class PDTB(object):
    def __init__(self, config):
        self.config = config
        self.word_to_id = {'<pad>':0, '</s>':1}
        self.word_cnt = {}
        self.vocab_size = 2
        
    def load_PDTB(self, mode):
        with open(self.config.resourses.data_base_dir + mode +"_pdtb.json", "r") as f:
            lines = f.readlines()

        data = [json.loads(line) for line in lines]

        arg1_sents = []
        arg2_sents = []
        labels = []

        for value in data:
            label_list = value['Sense']
            for label in label_list:
                label = label.split('.')[0]

                if label == "Comparison":
                    label = 0
                elif label == "Contingency":
                    label = 1
                elif label == "Expansion":
                    label = 2
                elif label == "Temporal":
                    label = 3
                else:
                    continue
                
                arg1_words = []
                for w in value['Arg1']['Word']:
                    arg1_words.append(w)
                    if not self.word_cnt.get(w):
                        self.word_cnt[w] = 0
                    self.word_cnt[w] += 1

                arg2_words = []
                for w in value['Arg2']['Word']:
                    arg2_words.append(w)
                    if not self.word_cnt.get(w):
                        self.word_cnt[w] = 0
                    self.word_cnt[w] += 1

                arg1_sents.append(arg1_words)
                arg2_sents.append(arg2_words)
                labels.append(label)
        
        assert(len(arg1_sents) == len(arg2_sents) == len(labels))
        
        return arg1_sents, arg2_sents, labels


    def build_vocab(self):
        self.word_cnt = sorted(self.word_cnt.items(), key = lambda x:int(x[1]), reverse=True)
        self.word_cnt = self.word_cnt[:self.config.model.top_words]

        # build up word dict
        for key, _ in self.word_cnt:
            self.word_to_id[key] = self.vocab_size
            self.vocab_size += 1
        
        assert(self.config.model.vocab_size == len(self.word_to_id))

        return self.word_to_id


def sent_to_tensor(batch, word_to_id, max_seq_len):
    '''
    Inputs:
        batch: [B * T]   type:string list
    Outpus:
        tensor: [T * B]  type:tensor
    '''
    batch_size = len(batch)
    
    tensor = torch.zeros(max_seq_len, batch_size, dtype=torch.long)
    for i in range(batch_size):
        min_len = min(len(batch[i], max_seq_len))
        for j in range(min_len):
            id = word_to_id.get(batch[i][j], 0)
            tensor[j][i] = id

    return tensor
    

