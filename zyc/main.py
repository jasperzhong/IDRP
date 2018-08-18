
import argparse

from train import train 
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help="train test or eval")

args = parser.parse_args()

config = Config()


if args.mode == "train":
    train(config)
elif args.mode == "test":
    pass
elif args.mode == "eval":
    pass
else:
    raise RuntimeError("Mode %s is undefined." % args.mode)