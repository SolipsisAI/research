from typing import List, Union, Dict, Tuple

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
    AutoModelForCausalLM
)
from datasets import Dataset, list_metrics, load_metric


def load_csv(filename):
    return pd.read_csv(filename)


def prepare_data(
    data_or_filename: Union[str, pd.DataFrame],
    filter_by: str = None,
    filter_value: str = None,
    content_key: str = "text",
    n: int = 7,
    test_size: float = 0.1,
    flatten: bool = True,
):
    data = load_csv(data_or_filename) if isinstance(data_or_filename, str) else data_or_filename

    contexted_data = prepare_context(
        data,
        filter_by=filter_by,
        filter_value=filter_value,
        content_key=content_key,
        n=n,
    )

    trn_df, val_df = train_test_split(contexted_data, test_size=test_size)
    
    if flatten:
        train_dataset = prepare_dataset(trn_df)
        val_dataset = prepare_dataset(val_df)
        return train_dataset, val_dataset

    return trn_df, val_df


def prepare_context(
    data: pd.DataFrame,
    filter_by: str = None,
    filter_value: str = None,
    content_key: str = "text",
    n: int = 7,
):
    if filter_by:
        indexes = data.loc[data[filter_by] == filter_value].index
    else:
        indexes = range(n, len(data[content_key]))

    contexted = []

    for i in indexes:
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(data.iloc[j][content_key])
        contexted.append(row)

    columns = ["response", "context"]
    columns = columns + ["context/" + str(i) for i in range(n - 1)]

    df = pd.DataFrame.from_records(contexted, columns=columns)
    return df


def prepare_dataset(df):
    columns = [col for col in df] 
    dataset = Dataset.from_pandas(concat_text(df))
    dataset = dataset.remove_columns(columns + ['__index_level_0__'])
    return dataset


def preprocess_function(tokenizer, max_length=256):
    def _construct(examples):
        flatten = lambda l: [item for sublist in l for item in sublist] 
        concat_text = f"{tokenizer.eos_token}".join(reversed([v for _, v in examples.items() if isinstance(v, str)]))
        concat_text = concat_text + tokenizer.eos_token
        tokenized = tokenizer(concat_text, padding="max_length",  max_length=max_length)
        examples["input_ids"] = tokenized["input_ids"]
        examples["attention_mask"] = tokenized["attention_mask"]
        return examples
        
    return _construct




def compute_metrics(eval_pred):
    metric = load_metric("perplexity")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



def train():
    MODEL_NAME = "microsoft/DialoGPT-small"
    model_cls = AutoModelForCausalLM
    tokenizer_cls = AutoTokenizer


    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  

    device = torch.device(dev) 

    print(f"""
        cuda: {torch.cuda.is_available()}
        current_device: {torch.cuda.current_device()}
        device_count: {torch.cuda.device_count()}
    """)


    config = AutoConfig.from_pretrained(MODEL_NAME)

    filepath = "data/processed.csv"
    df = pd.read_csv(filepath, encoding="utf-8", usecols=["character", "content"]).rename(columns={"content": "text"})

    pd.set_option("display.max_colwidth", None)
    df.tail(10)

    base_model = model_cls.from_pretrained(MODEL_NAME, config=config)
    base_tokenizer = tokenizer_cls.from_pretrained(MODEL_NAME)
    base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    train_df, val_df = prepare_data(df, filter_by="character", filter_value="bitjockey", flatten=False)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    tokenized_train_dataset = train_dataset.map(preprocess_function(tokenizer=base_tokenizer, max_length=256), remove_columns=list(train_dataset.features.keys()))
    tokenized_val_dataset = val_dataset.map(preprocess_function(tokenizer=base_tokenizer, max_length=256), remove_columns=list(val_dataset.features.keys()))
    
    tokenized_train_dataset.set_format(type="torch", columns=["input_ids"])
    tokenized_val_dataset.set_format(type="torch", columns=["input_ids"])

    FINETUNED_MODEL = 'SP-05162022a-myDialoGPT2-small'

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    trainer = None
    training_args = TrainingArguments(
        output_dir=FINETUNED_MODEL,          # output directory
        evaluation_strategy="epoch",
        num_train_epochs=3,           # total # of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,   # batch size for evaluation
        weight_decay=0.01,           # strength of weight decay
        logging_dir=FINETUNED_MODEL,            # directory for storing logs
        prediction_loss_only=True,
    )
    
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    
if __name__ == "__main__":
    train()