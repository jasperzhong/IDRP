"""
training
"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support as score

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
                config.model.n_layers)
   
    pdtb = PDTB(config)
    train_arg1_sents, train_arg2_sents, train_labels = pdtb.load_PDTB("train")
    dev_arg1_sents, dev_arg2_sents, dev_labels = pdtb.load_PDTB("dev")
    word_to_id = pdtb.build_vocab()
    model.to(device)
    
    start = time.time()
    model.load_pretrained_embedding(config.training.fix_embed, config.resourses.glove_path, word_to_id)
    print("Loading embedding taking %.3f s" % (time.time() - start))

    
    
    batch_size = config.training.batch_size
    max_seq_len = config.model.max_seq_len
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.training.lr, 
                weight_decay=config.training.weight_decay) # L2

    print("Start training!")
    best_f1 = 0.0
    for epoch in range(config.training.epochs):
        total_loss = 0.0
        start = time.time()

        result = []
        # train
        for i in range(0, len(train_arg1_sents), batch_size):
            optimizer.zero_grad()

            arg1 = train_arg1_sents[i: i + batch_size]
            arg2 = train_arg2_sents[i: i + batch_size]
            label = train_labels[i: i + batch_size]
            
            arg1 = sent_to_tensor(arg1, word_to_id, max_seq_len).to(device)
            arg2 = sent_to_tensor(arg2, word_to_id, max_seq_len).to(device)
            label = torch.LongTensor(label).to(device)

            output = model(arg1, arg2)
            result.extend(list(torch.max(output, 1)[1].cpu().numpy())) 

            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        precision, recall, f1, _  = score(train_labels, result, average='binary')
        print("Epoch %d: train f1 score: %.2f  precision: %.2f  recall: %.2f" % (epoch, 100 * f1, 
            100 * precision, 100 * recall))
        print("Epoch %d train loss: %.3f  time: %.3f s" % (epoch, total_loss / len(train_arg1_sents), time.time() - start))

        # dev
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

        # F1 score
        precision, recall, f1, _  = score(dev_labels, result, average='binary')
        print("Epoch %d: dev f1 score: %.2f  precision: %.2f  recall: %.2f" % (epoch, 100 * f1, 
            100 * precision, 100 * recall))
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model, config.resourses.model_path + config.type + "_" + 
                       config.resourses.model_name)
            print("Model saved!")
