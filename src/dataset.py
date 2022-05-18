from pathlib import Path

import pandas as pd

from datasets import Dataset, DatasetDict, list_metrics, load_metric, load_from_disk


def load_and_preprocess_datasets(data_dir, tokenizer):
    data = load_data(data_dir)
    datasets = create_datasets(data)
    preprocessed_datasets = preprocess_datasets(datasets, tokenizer)
    return DatasetDict(preprocessed_datasets)


def find_data_filepaths(data_dir):
    data_filepaths = list(Path(data_dir).glob("*.csv"))
    return data_filepaths


def load_data(data_dir):
    data = {}
    data_filepaths = find_data_filepaths(data_dir)
    for data_filepath in data_filepaths:
        data_name = data_filepath.stem
        data[data_name] = pd.read_csv(data_filepath, encoding="utf-8", on_bad_lines='skip')
    return data


def create_datasets(data, eos_token="<|endofsentence|>"):
    datasets = {}
    for name, df in data.items():
        grouped = df[["conv_id", "prompt", "utterance"]].groupby("conv_id")["utterance"]
        concat_text = grouped.transform(lambda x: eos_token.join(x))
        datasets[name] = Dataset.from_dict({"text": concat_text.unique()})
    return datasets


def preprocess_function(tokenizer, text_column="text", max_length=256):  
    def _tokenize(examples):
        flatten = lambda l: [item for sublist in l for item in sublist] 
        sanitized_text = [v.replace("_comma_", ",") for k, v in examples.items()]
        tokenized = tokenizer(
            sanitized_text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        examples["input_ids"] = flatten(tokenized["input_ids"])
        return examples
    return _tokenize


def preprocess_datasets(datasets, tokenizer, text_column="text", max_length=256):
    columns = lambda d: d.features.keys()
    preprocessed_datasets = {}
    
    for name, dataset in datasets.items():
        ds = dataset.map(
            preprocess_function(tokenizer, text_column, max_length),
            remove_columns=columns(dataset)
        )
        ds.set_format(type="torch", columns=["input_ids"])
        preprocessed_datasets[name] = ds
        
    return preprocessed_datasets