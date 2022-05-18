import argparse

from src.train import run
from src.args import Args
from src.utils import build_args


def main():
    default_args = Args().__dict__
    args = build_args(default_args)
    run(args)
