import argparse

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


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

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=512,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1] :][0],
        skip_special_tokens=True,
    )

    return response, chat_history_ids, step + 1


def chat(model, tokenizer):
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
            text=text,
            chat_history_ids=chat_history_ids,
            step=step,
        )
        print(f"Bot: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "-m")
    parser.add_argument("--tokenizer", "-t")

    args = parser.parse_args()

    finetuned_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    chat(finetuned_model, tokenizer)
