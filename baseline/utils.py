"""
utilities
"""
import json
import random
import string
from collections import Counter

import torch


class PDTB(object):
    def __init__(self, config):
        self.config = config
        self.word_to_id = {'<pad>':0, '</s>':1}  
        self.word_cnt = {}
        self.vocab_size = 2
        
    def load_PDTB(self, mode):
        if mode == "train":
            mode += "_sec_02_20"
        elif mode == "dev":
            mode += "_sec_00_01"
        elif mode == "test":
            mode += "_sec_21_22"

        # load json file
        with open(self.config.resourses.data_base_dir + self.config.type + "_vs_others/" +
            mode, "r") as f:
            lines = f.readlines()

        data = [json.loads(line) for line in lines]

        arg1_sents = []
        arg2_sents = []
        labels = []

        # extract instances
        for value in data:
            label_list = value['Sense']
            for label in label_list:
                label = label.split('.')[0]
                
                if self.config.type == "Ent+Exp":
                    if label == "Expansion" or label == "EntRel->Expansion":
                        label = 1
                    else:
                        label = 0
                else:
                    if label == self.config.type:
                        label = 1
                    else:
                        label = 0
                
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
        
        # shuffle
        c = list(zip(arg1_sents, arg2_sents, labels))
        random.shuffle(c)
        arg1_sents, arg2_sents, labels = zip(*c)

        return arg1_sents, arg2_sents, labels


    def build_vocab(self):
        '''
        build up vocabulary
        '''
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
        min_len = min(len(batch[i]), max_seq_len)
        for j in range(min_len):
            id = word_to_id.get(batch[i][j], 0)
            tensor[j][i] = id

    return tensor
