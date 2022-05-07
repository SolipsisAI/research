from typing import Union
from urllib import response

import pandas as pd

from sklearn.model_selection import train_test_split


def load_csv(filename):
    return pd.read_csv(filename)


def prepare_data(
    data_or_filename: Union[str, pd.DataFrame],
    filter_by: str = None,
    filter_value: str = None,
    content_key: str = "content",
    n: int = 7,
    test_size: float = 0.1,
):
    if isinstance(str):
        data = load_csv(data_or_filename)

    contexted_data = prepare_context(
        data,
        filter_by=filter_by,
        filter_value=filter_value,
        content_key=content_key,
        n=n,
    )

    trn_df, val_df = train_test_split(contexted_data, test_size=test_size)

    return {"train": trn_df, "validation": val_df}


def prepare_context(
    data: pd.DataFrame,
    filter_by: str = None,
    filter_value: str = None,
    content_key="content",
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
