import argparse

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.utils import export_model, make_tarfile


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", "-m")
    parser.add_argument("--tokenizer_path", "-t")
    parser.add_argument("--config_path", "-c")
    parser.add_argument("--output_path", "-o")

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = AutoConfig.from_pretrained(args.config_path)

    export_model(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_path=args.output_path,
    )
