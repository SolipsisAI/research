import argparse
import re
from datetime import datetime

from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Conversation,
    ConversationalPipeline,
)

from src.classifier import Classifier


def chat(model, tokenizer, output_dir, classifier=None, device="cuda", max_length: int = None):
    """Use model.generate to interact"""

    if max_length is None:
        max_length = model.config.max_length

    model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_ids = tokenizer.pad_token_id
    model.to(device)

    output_filepath = Path(output_dir).joinpath(f"chatlog-{datetime.now().isoformat()}.txt")
    with open(output_filepath, "w+") as chatlog:
        step = 0
        while True:
            text = input(">> ")

            if not text:
                continue

            if text in ["/q", "/quit", "/e", "/exit"]:
                break

            print(f"User: {text}")

            text = preprocess_text(text, classifier=classifier)
            chatlog.write("User: " + text + "\n")

            new_user_input_ids = tokenizer.encode(
                text + tokenizer.eos_token,
                return_tensors="pt",
            )

            # append the new user input tokens to the chat history
            bot_input_ids = (
                torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                if step > 0
                else new_user_input_ids
            )

            # generate chat ids
            chat_history_ids = model.generate(
                bot_input_ids,
                max_length=max_length,
                # Other args are set in the model.config
            )

            response = tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )

            chatlog.write("Bot:" + response + "\n")
            response = postprocess_text(response, classifier=classifier)

            print(f"Bot: {response}")


def chat_pipeline(model, tokenizer, output_dir, classifier=None, device="cuda", max_length=1000):
    pipe = ConversationalPipeline(
        model=model,
        tokenizer=tokenizer,
        device=-1 if device == "cpu" else device,
    )

    # Override the max_length. Other config is set in the model itself.
    pipe.model.config.max_length = max_length
    
    output_filepath = Path(output_dir).joinpath(f"chatlog-{datetime.now().isoformat()}.txt")
    with open(output_filepath, "w+") as chatlog:
        while True:
            text = input(">> ")
            if text in ["/q", "/quit", "/e", "/exit"]:
                break

            chatlog.write("User: " + text + "\n")
            conversation = Conversation(preprocess_text(text, classifier=classifier))

            print(f"User: {text}")

            result = pipe(conversation)
            response = result.generated_responses[-1]

            chatlog.write("Bot: " + response + "\n")

            postprocessed_text = postprocess_text(response, classifier=classifier)

            print(f"Bot: {postprocessed_text}")


def preprocess_text(text, classifier=None):
    """Prepend context label if classifier specified"""
    prefix = ""
    if classifier:
        context_label = classifier.classify(text, k=1)[0]
        prefix = f"{context_label} "
    return f"{prefix}{text}"


def postprocess_text(text, classifier=None):
    """Clean response text"""
    if classifier:
        text = re.sub(r"^\w+\s", "", text)
    return re.sub(r"_comma_", ",", text)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--model_name", "-m", required=True)
    parser.add_argument("--tokenizer", "-t", default=None)
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--classifier", "-cf", default=None)
    parser.add_argument("--pipeline", "-p", action="store_true", default=False)
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument("--max_length", default=1000)

    args = parser.parse_args()

    classifier = None

    if args.classifier is not None:
        classifier = Classifier(model=args.classifier)

    if not args.config:
        args.config = args.model_name

    if not args.tokenizer:
        args.tokenizer = args.model_name

    config = AutoConfig.from_pretrained(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        from_tf=False,
        config=config,
    )

    chat_fn = chat_pipeline if args.pipeline else chat

    print(f"Using device {args.device}")

    chat_fn(
        model,
        tokenizer,
        args.output_dir,
        device=args.device, 
        classifier=classifier,
        max_length=args.max_length,
    )
