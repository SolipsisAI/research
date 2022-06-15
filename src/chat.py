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


def chat(model, tokenizer, device, classifier=None):
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
            max_length=512,
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

        print(f"Bot: {clean_text(response)}")


def chat_pipeline(model, tokenizer, classifier=None, device=None):
    conversation = Conversation()
    pipe = ConversationalPipeline(
        model=model,
        tokenizer=tokenizer,
        device=-1 if device == "cpu" else device,
        max_length=tokenizer.max_len_single_sentence,
    )
    pipe.model.config.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.model.max_length = 128

    while True:
        text = input(">> ")
        if text in ["/q", "/quit", "/e", "/exit"]:
            break

        conversation.add_user_input(preprocess_text(text, classifier=classifier))

        print(f"User: {text}")

        result = pipe(conversation)
        response = result.generated_responses[-1]

        print(f"Bot: {clean_text(response)}")


def preprocess_text(text, classifier=None):
    """Prepend context label if classifier specified"""
    prefix = ""
    if classifier:
        context_label = classifier.classify(text, k=1)[0]
        prefix = f"{context_label} "
    return f"{prefix}{text}"


def clean_text(text):
    """Clean response text"""
    return re.sub(r"^\w+", "", text)


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

    chat_fn(model, tokenizer, classifier=classifier, device=args.device)
