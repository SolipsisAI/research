import json
from typing import Dict, List, Union

from transformers import PreTrainedModel, pipeline

from src.utils import ID2LABEL_FILEPATH, LABEL2ID_FILEPATH


def read_json(filename: str, as_type=None) -> Dict:
    with open(filename, "r") as json_file:
        data = json.load(json_file)

        if as_type is not None:
            data = dict([(as_type(k), v) for k, v in data.items()])

        return data


class Classifier:
    def __init__(
        self, model: Union[str, PreTrainedModel], id2label_file=None, label2id_file=None
    ) -> None:
        if not id2label_file:
            id2label_file = ID2LABEL_FILEPATH
        if not label2id_file:
            label2id_file = LABEL2ID_FILEPATH
        self.pipe = pipeline("text-classification", model=model, return_all_scores=True)
        self.pipe.model.config.id2label = read_json(id2label_file, as_type=int)
        self.pipe.model.config.label2id = read_json(label2id_file)

    def classify(self, text, k: int = 1) -> Union[List[Dict], List[str]]:
        results = self.pipe(text)
        
        if not results:
            return
        
        results.sort(key=lambda item: item.get("score"), reverse=True)

        # Return all the results
        if k < 1:
            return results

        # Sort by score, in descending order
        results.sort(key=lambda item: item.get("score"), reverse=True)

        # Return the top k results
        return [r["label"] for r in results[:k]]
