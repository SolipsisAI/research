import logging
import os
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import MODEL_WITH_LM_HEAD_MAPPING, PreTrainedTokenizer

from src.utils import clean_text

try:
    pass
except ImportError:
    pass

# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def construct_conv(row, tokenizer, eos=True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(
        reversed(
            [
                tokenizer.encode(clean_text(x, replace_comma=True))
                + [tokenizer.eos_token_id]
                for x in row
            ]
        )
    )
    conv = flatten(conv)
    return conv


class ConversationDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        args,
        df: pd.DataFrame,
        block_size: int = 512,
    ):
        block_size = block_size - (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )

        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = construct_conv(row, tokenizer)
                if len(conv) > block_size:
                    continue
                self.examples.append(conv)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)
