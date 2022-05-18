import argparse

from copy import copy

from transformers import TrainingArguments


class ArgBuilder:
    def __init__(self, required=None):
        if required is None:
            required = ["output_dir", "data_dir", "base_model", "text_column"]
        self._required = required
        self._parser = argparse.ArgumentParser()
        self.data_dir = None
        self.data_filepath = None
        self.base_model = None
        self.text_column = None
        self.group_column = None
        self.filter_by = None
        self.training_args = TrainingArguments(output_dir="an/example/here")

    def for_training(self):
        return self.training_args

    def build_and_parse(self):
        self.build()
        return self.parse()

    def build(self):
        run_args = copy(self.__dict__)
        training_args = run_args.pop(
            "_training_args", TrainingArguments(output_dir="an/example/here").__dict__
        )

        for args in [run_args, training_args]:
            self._build_args(args)

    def parse(self):
        return self._parser.parse_args()

    def _build_args(self, args, skip=None):
        for arg, val in args.items():
            val_type = type(val)
            options = {"default": val}

            if arg.startswith("_"):
                continue

            if arg in self._required:
                options["required"] = True
                options.pop("default")

            if val_type == bool:
                options["action"] = "store_true"

            self._parser.add_argument(f"--{arg}", **options)

        return self._parser

    def set_training_args(self, args):
        for arg, val in args.__dict__.items():
            if arg in self.training_args.__dict__:
                setattr(self.training_args, arg, val)
