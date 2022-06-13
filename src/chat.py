import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Conversation, pipeline

from src.classifier import Classifier
from src.utils import PAD_TOKEN


## Chat App
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
        top_p=0.7,
        #temperature=0.8,
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1] :][0],
        skip_special_tokens=True,
    )

    return response, chat_history_ids, step + 1


def chat(model, tokenizer, classifier=None):
    step = 0
    chat_history_ids = []

    while True:
        text = input(">> ")
        if text in ["/q", "/quit", "/e", "/exit"]:
            break

        print(f"User: {text}")

        prefix = ""

        if classifier:
            context_label = classifier.classify(text, k=1)[0]
            prefix = f"{context_label} "

        response, chat_history_ids, step = generate_responses(
            model=model,
            tokenizer=tokenizer,
            text=f"{prefix}text",
            chat_history_ids=chat_history_ids,
            step=step,
        )
        print(f"Bot: {response}")


def chat_pipeline(model, tokenizer):
    pipe = pipeline("conversational", model=model, tokenizer=tokenizer)
    # Disable the "Setting pad_token_id" message
    # https://github.com/huggingface/transformers/issues/12020#issuecomment-898899723
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

    conversation = None

    while True:
        text = input(">> ")
        if text in ["/q", "/quit", "/e", "/exit"]:
            break

        if not conversation:
            conversation = Conversation()

        conversation.add_user_input(text)

        print(f"User: {text}")
        result = pipe(conversation)
        response = result.generated_responses[-1]
        print(f"Bot: {response}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "-m")
    parser.add_argument("--tokenizer", "-t")
    parser.add_argument("--classifier", "-c", default=None)
    parser.add_argument("--pipeline", "-p", action="store_true", default=False)

    args = parser.parse_args()

    classifier = None

    if args.classifier is not None:
        classifier = Classifier(model=args.classifier)

    finetuned_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, pad_token=PAD_TOKEN)
    chat_fn = chat_pipeline if args.pipeline else chat

    chat_fn(finetuned_model, tokenizer, classifier=classifier)
