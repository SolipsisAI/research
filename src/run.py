import argparse

from args import Args
from train import run
from utils import build_args

if __name__ == "__main__":
    default_args = Args().__dict__
    args = build_args(default_args)
    run(args)
