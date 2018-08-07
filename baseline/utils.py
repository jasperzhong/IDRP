
import json
import random
import string

import torch

from config import config

def load_stopwords():
    with open(config.resourses.stop_word_path, "r") as f:
        stop_words = f.readlines()

    stop_words = [word.strip('\n') for word in stop_words]
    stop_words.extend(string.punctuation)

    return stop_words


def load_PDTB(type, classes, word_cnt={}):
    with open(config.resourses.data_base_dir + "im"+ type +"Corenlp_" + classes, "r") as f:
        data = json.load(f)
    
    stop_words = []
    max_seq_len = config.model.seq_len

    arg1_sents = []
    arg2_sents = []
    labels = []

    for _ ,value in data.items():
        label = value['Sense'].split('.')[0]
        if label == classes:
            label = 1
        else:
            label = 0
    
        arg1_words = []
        for word in value['Arg1']['Tokens']:
            if not stop_words or word['Word'] not in stop_words:
                arg1_words.append(word['Word'])
                if not word_cnt.get(word['Word']):
                    word_cnt[word['Word']] = 0
                word_cnt[word['Word']] += 1

        arg2_words = []
        for word in value['Arg2']['Tokens']:
            if not stop_words or word['Word'] not in stop_words:
                arg2_words.append(word['Word'])
                if not word_cnt.get(word['Word']):
                    word_cnt[word['Word']] = 0
                word_cnt[word['Word']] += 1


        if len(arg1_words) < max_seq_len and len(arg2_words) < max_seq_len:
            arg1_sents.append(arg1_words)
            arg2_sents.append(arg2_words)
            labels.append(label)
        # loss samples: train(432)  test(34)  dev(36)

    return arg1_sents, arg2_sents, labels, word_cnt



def build_up_word_dict(classes):
    word_to_id = {'OOV':0}
    num_words = 1
    word_cnt = {}
    _, _, _, word_cnt = load_PDTB("Train",classes, word_cnt)
    _, _, _, word_cnt = load_PDTB("Test",classes, word_cnt)
    _, _, _, word_cnt = load_PDTB("Dev",classes, word_cnt)

    # first 10k words
    word_cnt = sorted(word_cnt.items(), key = lambda x:int(x[1]), reverse=True)
    word_cnt = word_cnt[:config.model.top_words]

    # build up word dict
    for key, _ in word_cnt:
        word_to_id[key] = num_words
        num_words += 1
    config.model.vocab_size = len(word_to_id) + 1
    return word_to_id

def data_loader(type,classes, batch_size):
    arg1_sents, arg2_sents, labels, _ = load_PDTB(type,classes)
    print(len(arg1_sents))
    for i in range(0, len(arg1_sents), batch_size):
        yield (arg1_sents[i:i+batch_size], 
               arg2_sents[i:i+batch_size], 
               labels[i:i+batch_size])
    raise StopIteration


def sent_to_tensor(batch, word_to_id):
    '''
    Inputs:
        batch: [B * T]   type:string list
    Outpus:
        tensor: [T * B]  type:tensor
    '''
    pad_len = config.model.seq_len
    batch_size = len(batch)
    
    tensor = torch.zeros(pad_len, batch_size, dtype=torch.long)
    for i in range(batch_size):
        for j in range(len(batch[i])):
            id = word_to_id.get(batch[i][j], 0)
            tensor[j][i] = id

    return tensor
    

