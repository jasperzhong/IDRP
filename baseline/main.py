"""
main func
"""
import argparse

from config import Config
from train import train
from test import test

config = Config()

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help="train test or eval")
parser.add_argument('-type', type=str, help="four classes", choices=config.types)

parser.add_argument('-lr', type=float, help="learning rate", default=config.training.lr)
parser.add_argument('-batch', type=int, help="batch size", default=config.training.batch_size)
parser.add_argument('-l2', type=float, help="l2 regularization", default=config.training.weight_decay)
parser.add_argument('fix_embed', type=bool, help="whether fix embedding when training", default=True)

parser.add_argument('-hidden', type=int, help="hidden size", defualt=config.model.hidden_size)
parser.add_argument('-n_layers', type=int, help="number of stacked lstm", default=config.model.n_layers)

parser.add_argument('-model_name', type=str, help="model name", default=config.resourses.model_name)

args = parser.parse_args()

config.type = args.type

config.training.lr = args.lr
config.training.batch_size = args.batch
config.training.weight_decay = args.l2 
config.training.fix_embed = args.fix_embed

config.model.hidden_size = args.hidden
config.model.n_layers = args.n_layers

config.resourses.model_name = args.model_name

if args.mode == "train":
    train(config)
elif args.mode == "test":
    test(config)
elif args.mode == "eval":
    pass
else:
    raise RuntimeError("Mode %s is undefined." % args.mode)
