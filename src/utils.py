from typing import Union

import pandas as pd

from sklearn.model_selection import train_test_split


def prepare_data(data: Union[pd.DataFrame, str], text_key="content"):
    """Prepare and split data into training and evaluation sets"""
    if isinstance(data, str):
        data = pd.read_csv(data)

    contexted = []

    n = 7

    for i in range(n, len(data[text_key])):
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(data[text_key][j])
        contexted.append(row)

    columns = ["response", "context"]
    columns = columns + ["context/" + str(i) for i in range(n - 1)]
    trn_df, val_df = train_test_split(df, test_size=0.1)

    return trn_df, val_df
