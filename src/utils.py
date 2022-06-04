from sklearn.model_selection import train_test_split


def prepare_data(data, text_key="content"):
    """Prepare data"""
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
