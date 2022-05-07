import argparse

from train import run
from args import Args


if __name__ == "__main__":
    default_args = Args().__dict__

    parser = argparse.ArgumentParser()
    
    for arg, val in default_args.items():
        val_type = type(val)
        flag = f"--{arg}"

        if val_type == bool:
            parser.add_argument(flag, action="store_true", default=val)
            continue

        parser.add_argument(flag, type=type(val), default=val)

    args = parser.parse_args()
    
    run(args)
