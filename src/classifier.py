from typing import Union

from transformers import PreTrainedModel, pipeline

from src.utils import DATA_DIR, read_json

ID2LABEL_FILE = DATA_DIR.joinpath("id2label.json")
LABEL2ID_FILE = DATA_DIR.joinpath("label2id.json")


class Classifier:
    def __init__(self, model: Union[str, PreTrainedModel]) -> None:
        self.pipe = pipeline("text-classification", model=model, return_all_scores=True)
        self.pipe.model.config.id2label = read_json(ID2LABEL_FILE)
        self.pipe.model.config.label2id = read_json(LABEL2ID_FILE)

    def classify(self, text):
        results = self.pipe(text)
        max_score = max(results[0], key=lambda x: x["score"])
        return max_score["label"]
