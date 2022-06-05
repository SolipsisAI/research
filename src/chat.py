import argparse

from transformers import pipeline, Conversation


def chat(model, tokenizer):
    conversational_pipeline = pipeline(
        "conversational", model=model, tokenizer=tokenizer)
    
    conversation_map = {}

    while True:
        text = input(">> ")

        if text in ["/q", "/quit", "/e", "/exit"]:
            break

        if not conversation_map:
            conversation = Conversation()
            conversation_map[conversation.uuid] = conversation
        
        conversation_map[conversation.uuid].add_user_input(text)
        print(f"User: {text}")

        conversational_pipeline(conversation) 
        print(f"Bot: {conversation.generated_responses[-1]}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-m", default="microsoft/DialoGPT-small")
    parser.add_argument("--tokenizer", "-t", default="microsoft/DialoGPT-small")

    args = parser.parse_args()

    chat(args.model, args.tokenizer)
