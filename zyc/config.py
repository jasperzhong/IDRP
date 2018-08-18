'''
some configurations 
'''

class ResourcesConfig(object):
    glove_path = "../data/glove.840B.300d.txt"
    stop_word_path = "../data/english.txt"
    data_base_dir = "../data/lin/"
    model_path = "../model/"
    model_name = "single.pkl"

class TrainingConfig(object):
    lr = 1e-4
    batch_size = 32
    epochs = 50
    weight_decay = 1e-5
    

class ModelConfig(object):
    max_seq_len = 100 
    embedd_size = 300
    vocab_size = 10002
    hidden_size = 128 
    r = 10
    d_a = 128

    top_words = 10000


class Config(object):
    resourses = ResourcesConfig()
    training = TrainingConfig()
    model = ModelConfig()

