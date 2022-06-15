import argparse
import re

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Conversation,
    ConversationalPipeline,
)

from src.classifier import Classifier


def chat(model, tokenizer, device, classifier=None, max_length: int = 1000):
    """Use model.generate to interact"""
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)

    step = 0

    while True:
        text = input(">> ")
        if text in ["/q", "/quit", "/e", "/exit"]:
            break

        print(f"User: {text}")

        new_user_input_ids = tokenizer.encode(
            preprocess_text(text, classifier=classifier) + tokenizer.eos_token,
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
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8,
        )

        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1] :][0],
            skip_special_tokens=True,
        )

        print(f"Bot: {postprocess_text(response)}")


def chat_pipeline(model, tokenizer, classifier=None, device=None, max_length=1000):
    conversation = Conversation()
    pipe = ConversationalPipeline(
        model=model,
        tokenizer=tokenizer,
        device=-1 if device == "cpu" else device,
    )

    # Set model configuration
    # TODO: Save this to file
    pipe.model.config.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.model.config.max_length = max_length
    pipe.model.config.no_repeat_ngram_size = 3
    pipe.model.config.do_sample = True
    pipe.model.config.top_k = 100
    pipe.model.config.top_p = 0.7
    pipe.model.config.temperature = 0.8

    while True:
        text = input(">> ")
        if text in ["/q", "/quit", "/e", "/exit"]:
            break

        conversation.add_user_input(preprocess_text(text, classifier=classifier))

        print(f"User: {text}")

        result = pipe(conversation)
        response = result.generated_responses[-1]

        print(f"Bot: {postprocess_text(response)}")


def preprocess_text(text, classifier=None):
    """Prepend context label if classifier specified"""
    prefix = ""
    if classifier:
        context_label = classifier.classify(text, k=1)[0]
        prefix = f"{context_label} "
    return f"{prefix}{text}"


def postprocess_text(text):
    """Clean response text"""
    text = re.sub(r"^\w+", "", text)
    return re.sub(r"_comma_", ",", text)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "-m")
    parser.add_argument("--tokenizer", "-t")
    parser.add_argument("--config", "-c")
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

    config = AutoConfig.from_pretrained(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        from_tf=False,
        config=config,
    )

    chat_fn = chat_pipeline if args.pipeline else chat

    chat_fn(model, tokenizer, classifier=classifier, device=args.device, max_length=args.max_length)
