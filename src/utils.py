import pandas as pd

from sklearn.model_selection import train_test_split


def prepare_context(data, character, content_key="content"):
    character_indexes = data.loc[data['character'] == character].index
    
    contexted = []
    n = 7
     
    for i in character_indexes:
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(data.iloc[j][content_key])
        contexted.append(row)
    
    columns = ["response", "context"]
    columns = columns + ["context/" + str(i) for i in range(n-1)]
    
    df = pd.DataFrame.from_records(contexted, columns=columns)
    return df