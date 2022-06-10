import argparse
import glob
import random
import re
import shutil
import tarfile
import os.path
import logging

from typing import Dict, List, Union
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

logger = logging.getLogger(__name__)


def sorted_checkpoints(
    args, checkpoint_prefix="checkpoint", use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix))
    )

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)


def load_csv(filename):
    return pd.read_csv(filename)


def build_args(default_args: Dict):
    parser = argparse.ArgumentParser()

    for arg, val in default_args.items():
        val_type = type(val)
        flag = f"--{arg}"

        if val_type == bool:
            parser.add_argument(flag, action="store_true", default=val)
            continue

        parser.add_argument(flag, default=val)

    return parser.parse_args()


def export_model(model_path, output_path):
    model = AutoModelWithLMHead.from_pretrained(model_path)
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.save_pretrained(output_path)
    make_tarfile(f"{output_path}.tar.gz", output_path)
    print(f"Saved to {output_path}")


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def prepare_data(
    data: Union[str, pd.DataFrame],
    filter_by: str,
    content_key: str = "content",
    n: int = 7,
    test_size: float = 0.1,
):
    if isinstance(data, str):
        data = load_csv(data)

    contexted_data = make_context(
        data,
        filter_by=filter_by,
        content_key=content_key,
    )

    trn_df, val_df = train_test_split(contexted_data, test_size=test_size)

    return trn_df, val_df


def make_context(data, filter_by, content_key="content"):
    filter_key, filter_value = None, None if not filter_by else filter_by.split("==")

    contexted = []
    n = 7

    if filter_by:
        indexes = data.loc[data[filter_key] == filter_value].index
    else:
        indexes = range(n, len(data[content_key]))

    for i in indexes:
        if filter_key and filter_value and i < n:
            continue

        row = []
        prev = (
            i - 1 - n
        )  # we additionally subtract 1, so row will contain current response and 7 previous responses
        for j in range(i, prev, -1):
            row.append(data[content_key][j])
        contexted.append(row)

    columns = ["response", "context"]
    columns = columns + ["context/" + str(i) for i in range(n - 1)]

    df = pd.DataFrame.from_records(contexted, columns=columns)
    return train_test_split(df, test_size=0.1)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
