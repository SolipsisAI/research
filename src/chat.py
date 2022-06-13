import argparse
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Conversation,
    pipeline,
    ConversationalPipeline,
)

from src.classifier import Classifier
from src.utils import PAD_TOKEN


def generate_responses(model, tokenizer, text, chat_history_ids=None, step=0):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        text + tokenizer.eos_token, return_tensors="pt"
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
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.9,
        temperature=0.7,
    )
    # chat_history_ids = model.generate(
    #     bot_input_ids,
    #     max_length=1024,
    #     pad_token_id=tokenizer.pad_token_id,
    # )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1] :][0],
        skip_special_tokens=True,
    )

    return response, chat_history_ids, step + 1


def chat(model, tokenizer, classifier=None, device=None):
    """Use model.generate to interact"""
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)

    step = 0
    chat_history_ids = []

    while True:
        text = input(">> ")
        if text in ["/q", "/quit", "/e", "/exit"]:
            break

        print(f"User: {text}")

        response, chat_history_ids, step = generate_responses(
            model=model,
            tokenizer=tokenizer,
            text=preprocess_text(text, classifier=classifier),
            chat_history_ids=chat_history_ids,
            step=step,
        )

        print(f"Bot: {clean_text(response)}")


# def chat_pipeline2(model, tokenizer, classifier=None, device=None):
#     """Use conversational pipeline to interact"""
#     pipe = pipeline(
#         "conversational",
#         model=model,
#         tokenizer=tokenizer,
#         min_length_for_response=20,
#         device=device,
#     )
#     # Disable the "Setting pad_token_id" message
#     # https://github.com/huggingface/transformers/issues/12020#issuecomment-898899723
#     pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id

#     conversation = None

#     while True:
#         text = input(">> ")
#         if text in ["/q", "/quit", "/e", "/exit"]:
#             break

#         if not conversation:
#             conversation = Conversation()

#         conversation.add_user_input(preprocess_text(text, classifier=classifier))

#         print(f"User: {text}")

#         result = pipe(conversation)
#         response = result.generated_responses[-1]

#         print(f"Bot: {clean_text(response)}")


def chat_pipeline(model, tokenizer, classifier=None, device=None):
    conversation = Conversation()
    pipe = ConversationalPipeline(
        model=model, tokenizer=tokenizer, device=-1 if device == "cpu" else device
    )
    pipe.model.config.pad_token_id = pipe.tokenizer.pad_token_id

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

    config.min_length = 2
    config.max_length = 1000

    print(f"min_length: {config.min_length}")
    print(f"max_length: {config.max_length}")

    # https://github.com/huggingface/transformers/issues/7800#issuecomment-709021207
    # config = AutoConfig.from_pretrained(args.model_name)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # model.to(args.device)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, pad_token=PAD_TOKEN)
    chat_fn = chat_pipeline if args.pipeline else chat

    chat_fn(model, tokenizer, classifier=classifier, device=args.device)
