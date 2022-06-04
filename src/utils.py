from typing import Union, Dict

import pandas as pd

from sklearn.model_selection import train_test_split


def prepare_data(
    data: Union[pd.DataFrame, str], text_key="content", filter_args: Dict = None
):
    """Prepare and split data into training and evaluation sets"""
    if isinstance(data, str):
        data = pd.read_csv(data)

    if filter_args:
        indexes = data.loc[data[filter_args["key"]] == filter_args["value"]].index
    else:
        indexes = range(n, len(data[text_key]))

    contexted = []

    n = 7

    for i in indexes:
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(data.iloc[j][text_key])
        contexted.append(row)

    columns = ["response", "context"]
    columns = columns + ["context/" + str(i) for i in range(n - 1)]

    df = pd.DataFrame.from_records(contexted, columns=columns)

    trn_df, val_df = train_test_split(df, test_size=0.1)

    return trn_df, val_df
