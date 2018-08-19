"""
testing
"""
import numpy as np
import torch
from sklearn.metrics import classification_report


def test(config):
    '''
    test 4-way result!
    '''

    # load models
    models = []
    for type in config.types:
        models.append(torch.load(config.resources.model_path + type + config.resources.model_name))

    # load test dataset 
    test_arg1_sents, test_arg2_sents, test_labels = pdtb.load_PDTB("test")

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

            outputs = []
            for i in range(4):
                outputs.append(model(arg1, arg2))
            
            for i in range(len(arg1)):
                result.append(torch.argmax(outputs[0][i], outputs[1][i], outputs[2][i], outputs[3][i]))

        print(classification_report(test_labels, result))
