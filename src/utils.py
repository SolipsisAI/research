from typing import Union

import pandas as pd

from sklearn.model_selection import train_test_split


def load_csv(filename):
    return pd.read_csv(filename)


def prepare_context(
    data_or_filename: Union[str, pd.DataFrame], character, content_key="content"
):
    if isinstance(str):
        data = load_csv(data_or_filename)

    character_indexes = data.loc[data["character"] == character].index

    contexted = []
    n = 7

    for i in character_indexes:
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(data.iloc[j][content_key])
        contexted.append(row)

    columns = ["response", "context"]
    columns = columns + ["context/" + str(i) for i in range(n - 1)]

    df = pd.DataFrame.from_records(contexted, columns=columns)
    return df
