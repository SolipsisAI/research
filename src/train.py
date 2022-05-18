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

from src.args import ArgBuilder
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
    config,
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
    train_dataset = preprocessed_datasets["train"]
    eval_dataset = preprocessed_datasets["valid"]

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


def main(args, training_args): 
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Setup base model
    config = AutoConfig.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config)
    base_model.resize_token_embeddings(len(tokenizer))

    preprocessed_datasets = load_and_preprocess_datasets(
        data_dir_or_filepath=args.data_dir or args.data_filepath,
        tokenizer=tokenizer,
        text_column=args.text_column,
        group_column=args.group_column,
        filter_by=args.filter_by,
    )

    train(
        base_model=base_model,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        config=config,
        preprocessed_datasets=preprocessed_datasets,
        training_args=training_args,
    )


if __name__ == "__main__":
    arg_builder = ArgBuilder()
    args = arg_builder.build_and_parse()
    arg_builder.set_training_args(args)
    
    main(args, arg_builder.training_args)
