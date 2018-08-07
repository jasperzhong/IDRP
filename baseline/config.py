'''
some configurations 
'''

class ResourcesConfig(object):
    glove_path = r"D:\Download\迅雷下载\glove.840B.300d\glove.840B.300d.txt"
    stop_word_path = "data/english.txt"
    data_base_dir = "data/"
    model_path = "model/"
    model_name = "single"

class TrainingConfig(object):
    lr = 1e-3
    batch_size = 128
    epochs = 50
    weight_decay = 1e-4
    

class ModelConfig(object):
    seq_len = 50  #need to compute... 
    embedd_size = 50
    vocab_size = 10001  #need to extract from glove... a little big
    hidden_size = 128 
    top_words = 10000


class Config(object):
    resourses = ResourcesConfig()
    training = TrainingConfig()
    model = ModelConfig()

config = Config()