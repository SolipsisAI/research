import argparse

from src.utils import export_model, make_tarfile


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", "-m")
    parser.add_argument("--tokenizer_path", "-t")
    parser.add_argument("--output_path", "-o")

    args = parser.parse_args()

    export_model(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
    )
