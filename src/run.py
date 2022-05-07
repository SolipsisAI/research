import argparse

from train import run
from args import Args
from utils import build_args


if __name__ == "__main__":
    default_args = Args().__dict__
    args = build_args(default_args)
    run(args)
