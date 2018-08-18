import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report

from config import config
from model import Model
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train():
    choise = "cuda" if torch.cuda.is_available() else "cpu"
    print(choise + " is available")
    device = torch.device(choise)

    try:
        model = torch.load(config.resourses.model_path + config.resourses.model_name)
    except FileNotFoundError:
        print("Training from scratch!")
        model = Model(config.model.vocab_size, 
                    config.model.embedd_size,
                    config.model.hidden_size,
                    config.model.seq_len,
                    config.model.d_a,
                    config.model.r,
                    n_layers=2)
    '''
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal(m.weight.data,mean=0,std=0.1)
            torch.nn.init.constant(m.bias.data, 0.1)'''
    #model.apply(weights_init)
    model.to(device)
    word_to_id = build_up_word_dict()
    batch_size = config.training.batch_size
    train_arg1_sents, train_arg2_sents, train_labels, _ = load_PDTB("Train")
    dev_arg1_sents, dev_arg2_sents, dev_labels, _ = load_PDTB("Dev")
    
    loss_func = nn.NLLLoss(torch.FloatTensor([7,4,2,21]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr, 
                weight_decay=config.training.weight_decay) # L2
    
    start = time.time()
    model.load_pretrained_embedding(word_to_id)
    print("Loading embedding taking %.3f s" % (time.time() - start))

    print("Start training!")
    for epoch in range(config.training.epochs):
        total_loss = 0.0
        cnt = 0
        start = time.time()
        for i in range(0, len(train_arg1_sents), batch_size):
            loss = 0.0
            optimizer.zero_grad()

            arg1 = train_arg1_sents[i:i+batch_size]
            arg2 = train_arg2_sents[i:i+batch_size]
            label = train_labels[i:i+batch_size]
            
            arg1 = sent_to_tensor(arg1, word_to_id).to(device)
            arg2 = sent_to_tensor(arg2, word_to_id).to(device)
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
                arg1 = dev_arg1_sents[i:i+batch_size]
                arg2 = dev_arg2_sents[i:i+batch_size]
                label = dev_labels[i:i+batch_size]
                
                arg1 = sent_to_tensor(arg1, word_to_id).to(device)
                arg2 = sent_to_tensor(arg2, word_to_id).to(device)
                label = torch.LongTensor(label).to(device)

                output = model(arg1, arg2)
                result.extend(list(torch.max(output, 1)[1].cpu().numpy())) 

            print(classification_report(dev_labels, result))


if __name__=="__main__":
    train()
