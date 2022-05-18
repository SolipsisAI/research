from typing import List, Union
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, list_metrics, load_metric, load_from_disk


def load_and_preprocess_datasets(data_dir, tokenizer, **kwargs):
    data = load_data(data_dir)
    datasets = create_datasets(data, **kwargs)
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
        data[data_name] = pd.read_csv(
            data_filepath, encoding="utf-8", on_bad_lines="skip"
        )
    return data


def create_datasets(
    data,
    text_column: str = None,
    group_column: str = None,
    filter_by: str = None,
    eos_token="<|endofsentence|>",
    n: int = 7,
    test_size: float = 0.1,
):
    filter_key = None
    filter_value = None

    if filter_by:
        filter_key, filter_value = filter_by.split(":=")

    should_group = bool(group_column)

    datasets = {}

    if isinstance(data, pd.DataFrame):
        trn_df, val_df = train_test_split(data, test_size=test_size, shuffle=False)
        data = {
            "train": trn_df,
            "valid": val_df,
        }

    for name, df in data.items():
        if should_group:
            concat_text = df[[group_column, text_column]].groupby(group_column)[text_column].transform(lambda x: eos_token.join(x)).unique()
        else:
            _data = prepare_context(
                data=df,
                filter_by=filter_key,
                filter_value=filter_value,
                content_key=text_column,
                n=n,
            )
            concat_text = _data["text"]
            
        datasets[name] = Dataset.from_dict({"text": concat_text})

    return datasets


def prepare_context(
    data: pd.DataFrame,
    filter_by: str = None,
    filter_value: str = None,
    text_column: str = "text",
    eos_token: str = "<|endofsentence|>",
    n: int = 7,
):
    if filter_by:
        indexes = data.loc[data[filter_by] == filter_value].index
        for idx, i in enumerate(indexes):
            if i > n:
                break
        indexes = indexes[idx:]
    else:
        indexes = range(n, len(data[text_column]))

    contexted = []

    for i in indexes:
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(data.iloc[j][text_column])
        concat_text = eos_token.join(reversed(row))
        contexted.append([concat_text])
        
    columns = ["text"]
    df = pd.DataFrame.from_records(contexted, columns=columns)

    return df


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
            remove_columns=columns(dataset),
        )
        ds.set_format(type="torch", columns=["input_ids"])
        preprocessed_datasets[name] = ds

    return preprocessed_datasets
