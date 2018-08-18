import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report

from config import Config
from model import Model
from utils import PDTB, sent_to_tensor


def train(config):
    choise = "cuda" if torch.cuda.is_available() else "cpu"
    print(choise + " is available")
    device = torch.device(choise)

    print("Training from scratch!")
    model = Model(config.model.vocab_size, 
                config.model.embedd_size,
                config.model.hidden_size,
                config.model.max_seq_len,
                config.model.d_a,
                config.model.r,
                n_layers=2)
    '''
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal(m.weight.data,mean=0,std=0.1)
            torch.nn.init.constant(m.bias.data, 0.1)'''
    #model.apply(weights_init)
    if torch.cuda.device_count() > 1:
        print("Multi-GPUs are available!")
        model = nn.DataParallel(model)
    
    model.to(device)

    pdtb = PDTB(config)
    train_arg1_sents, train_arg2_sents, train_labels = pdtb.load_PDTB("train")
    dev_arg1_sents, dev_arg2_sents, dev_labels = pdtb.load_PDTB("dev")
    word_to_id = pdtb.build_vocab()

    batch_size = config.training.batch_size
    max_seq_len = config.model.max_seq_len
    
    loss_func = nn.NLLLoss(torch.FloatTensor([7, 4, 2, 21]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr, 
                weight_decay=config.training.weight_decay) # L2
    
    start = time.time()
    model.load_pretrained_embedding(config.resourses.glove_path, word_to_id)
    print("Loading embedding taking %.3f s" % (time.time() - start))

    print("Start training!")
    for epoch in range(config.training.epochs):
        total_loss = 0.0
        cnt = 0
        start = time.time()
        for i in range(0, len(train_arg1_sents), batch_size):
            loss = 0.0
            optimizer.zero_grad()

            arg1 = train_arg1_sents[i: i + batch_size]
            arg2 = train_arg2_sents[i: i + batch_size]
            label = train_labels[i: i + batch_size]
            
            arg1 = sent_to_tensor(arg1, word_to_id, max_seq_len).to(device)
            arg2 = sent_to_tensor(arg2, word_to_id, max_seq_len).to(device)
            label = torch.LongTensor(label).to(device)

            output = model(arg1, arg2)
            loss += loss_func(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() / arg1.size(0)
            cnt += 1
        print("Epoch %d train loss: %.3f  time: %.3f s" % (epoch, total_loss / cnt, time.time() - start))

        # F1 score
        with torch.no_grad():
            result = []
            for i in range(0, len(dev_arg1_sents), batch_size):
                arg1 = dev_arg1_sents[i: i + batch_size]
                arg2 = dev_arg2_sents[i: i + batch_size]
                label = dev_labels[i: i + batch_size]
                
                arg1 = sent_to_tensor(arg1, word_to_id, max_seq_len).to(device)
                arg2 = sent_to_tensor(arg2, word_to_id, max_seq_len).to(device)
                label = torch.LongTensor(label).to(device)

                output = model(arg1, arg2)
                result.extend(list(torch.max(output, 1)[1].cpu().numpy())) 

            print(classification_report(dev_labels, result))


