import argparse

from typing import List, Dict, Union, Tuple

import torch, os, re, pandas as pd, json
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from datasets import Dataset, DatasetDict, list_metrics, load_metric, load_from_disk

from src.args import Args
from src.dataset import load_and_preprocess_datasets


def compute_metrics(eval_pred):
    metric = load_metric("perplexity")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(
    base_model,
    tokenizer,
    output_dir,
    preprocessed_datasets: DatasetDict,
    training_args: TrainingArguments,
):
    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    # Datasets
    train_dataset = preprocessed_datasets["valid"]
    eval_dataset = preprocessed_datasets["test"]

    # Setup trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Run training
    trainer.train()

    print("Training completed")

    # Save model, tokenizer, and config
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    print("Run the code below in a Jupyter cell")
    print(
        f"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        finetuned_model = AutoModelForCausalLM.from_pretrained({output_dir})
        tokenizer = AutoTokenizer.from_pretrained({output_dir})
        
        # Start the chat
        chat(finetuned_model, tokenizer)
        """
    )


def build_args(default_args: Dict):
    parser = argparse.ArgumentParser()

    for arg, val in default_args.items():
        val_type = type(val)
        flag = f"--{arg}"

        if val_type == bool:
            parser.add_argument(flag, action="store_true", default=val)
            continue

        parser.add_argument(flag, default=val)

    return parser.parse_args()


def main(args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        evaluation_strategy="epoch",
        num_train_epochs=args.epochs,  # total # of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        weight_decay=0.01,  # strength of weight decay
        logging_dir=args.output_dir,  # directory for storing logs
        prediction_loss_only=True,
    )

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Setup base model
    config = AutoConfig.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config)
    base_model.resize_token_embeddings(len(tokenizer))

    preprocessed_datasets = load_and_preprocess_datasets(args.data_dir, tokenizer)

    train(
        base_model=base_model,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        preprocessed_datasets=preprocessed_datasets,
        training_args=training_args,
    )


if __name__ == "__main__":
    default_args = Args().__dict__
    args = build_args(default_args)

    main(args)
