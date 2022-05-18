import argparse

from copy import copy

from transformers import TrainingArguments


class Args:
    def __init__(self, required=None):
        if required is None:
            required = ["output_dir", "data_dir", "base_model", "text_column"]
        self._required = required
        self.data_dir = None
        self.data_filepath = None
        self.base_model = None
        self.text_column = None
        self.group_column = None
        self.filter_by = None
        self.training_args = TrainingArguments(output_dir="an/example/here")

    def for_training(self):
        return self.training_args

    def build(self):
        parser = argparse.ArgumentParser()

        run_args = copy(self.__dict__)
        training_args = run_args.pop("_training_args", TrainingArguments(output_dir="an/example/here").__dict__) 
        
        for args in [run_args, training_args]:
            parser = self._build_args(parser, args)
        
        return parser.parse_args()
    
    def _build_args(self, parser, args, skip=None):
        for arg, val in args.items():
            val_type = type(val)
            params = {"default": val, "type": val_type}
            
            if arg.startswith("_"):
                continue
            
            if arg in self._required:
                params["required"] = True
                params.pop("default")

            if val_type == bool:
                params["action"] = "store_true"
                params.pop("type")

            parser.add_argument(f"--{arg}", **params)
        
        return parser