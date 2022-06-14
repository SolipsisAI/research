# SolipsisAI - research

This is a library used for training the models in our apps.

- [SolipsisAI - research](#solipsisai---research)
  - [Credits](#credits)
  - [Setup](#setup)
  - [Usage](#usage)

## Credits

The scripts/code here were heavily lifted from:

- [How to fine-tune the DialoGPT model on a new dataset for open-dialog conversational chatbots](https://github.com/ncoop57/i-am-a-nerd/blob/master/_notebooks/2020-05-12-chatbot-part-1.ipynb) by [Nathan Cooper](https://github.com/ncoop57)
- [Make your own Rick Sanchez (bot) with Transformers and DialoGPT fine-tuning](https://colab.research.google.com/drive/15wa925dj7jvdvrz8_z3vU7btqAFQLVlG) by [Rostyslav Neskorozhenyi](https://www.linkedin.com/in/slanj)

## Setup

```shell
pip install -e .
```

## Usage

```shell
solipsis-trainer --output_dir=../models/ERICA-medium \
    --data_filename=../data/empathetic_dialogue_processed.csv \
    --filter_by="speaker==<s1>" \
    --model_name_or_path="microsoft/DialoGPT-medium" \
    --config_name="microsoft/DialoGPT-medium" \
    --tokenizer_name="microsoft/DialoGPT-medium" \
    --content_key="text"
    --num_history=7
```