"""
testing
"""
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support as score

from utils import PDTB, sent_to_tensor

def test(config):
    choise = "cuda" if torch.cuda.is_available() else "cpu"
    print(choise + " is available")
    device = torch.device(choise)

    # load models
    model = torch.load(config.resourses.model_path + config.type + "_" + config.resourses.model_name).to(device)

    pdtb = PDTB(config)
    # load test dataset 
    train_arg1_sents, train_arg2_sents, train_labels = pdtb.load_PDTB("train")
    dev_arg1_sents, dev_arg2_sents, dev_labels = pdtb.load_PDTB("dev")
    test_arg1_sents, test_arg2_sents, test_labels = pdtb.load_PDTB("test")
    word_to_id = pdtb.build_vocab()

    batch_size = config.training.batch_size
    max_seq_len = config.model.max_seq_len

     # dev
    with torch.no_grad():
        result = []
        for i in range(0, len(test_arg1_sents), batch_size):
            arg1 = test_arg1_sents[i: i + batch_size]
            arg2 = test_arg2_sents[i: i + batch_size]
            label = test_labels[i: i + batch_size]
            
            arg1 = sent_to_tensor(arg1, word_to_id, max_seq_len).to(device)
            arg2 = sent_to_tensor(arg2, word_to_id, max_seq_len).to(device)
            label = torch.LongTensor(label).to(device)

            output = model(arg1, arg2)
            result.extend(list(torch.max(output, 1)[1].cpu().numpy())) 

        # F1 score
        f1, precision, recall, _  = score(test_labels, result, average='binary')
        print("f1 score: %.2f  precision: %.2f  recall: %.2f" % (100 * f1, 
            100 * precision, 100 * recall))

def ensemble_test(config):
    '''
    test 4-way result!
    '''
    choise = "cuda" if torch.cuda.is_available() else "cpu"
    print(choise + " is available")
    device = torch.device(choise)

    # load models
    models = []
    for type in config.types:
        models.append(torch.load(config.resourses.model_path + type + "_" + config.resourses.model_name).to(device))

    pdtb = PDTB(config)
    # load test dataset 
    train_arg1_sents, train_arg2_sents, train_labels = pdtb.load_PDTB("train")
    dev_arg1_sents, dev_arg2_sents, dev_labels = pdtb.load_PDTB("dev")
    test_arg1_sents, test_arg2_sents, test_labels = pdtb.load_PDTB("test")
    word_to_id = pdtb.build_vocab()

    batch_size = config.training.batch_size
    max_seq_len = config.model.max_seq_len

     # dev
    with torch.no_grad():
        result = []
        for i in range(0, len(test_arg1_sents), batch_size):
            arg1 = test_arg1_sents[i: i + batch_size]
            arg2 = test_arg2_sents[i: i + batch_size]
            label = test_labels[i: i + batch_size]
            
            arg1 = sent_to_tensor(arg1, word_to_id, max_seq_len).to(device)
            arg2 = sent_to_tensor(arg2, word_to_id, max_seq_len).to(device)
            label = torch.LongTensor(label).to(device)

            outputs = []
            for i in range(4):
                outputs.append(models[i](arg1, arg2))
            
            for i in range(len(arg1)):
                result.append(np.argmax([outputs[0][i][1], outputs[1][i][1], outputs[2][i][1], outputs[3][i][1]]))

        print(classification_report(test_labels, result))
