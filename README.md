# SolipsisAI - research

This is a library used for training the models in our apps.

- [SolipsisAI - research](#solipsisai---research)
  - [Credits](#credits)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Trainer](#trainer)
    - [Export](#export)
    - [Chat](#chat)

## Credits

The scripts/code here were heavily lifted from:

- [How to fine-tune the DialoGPT model on a new dataset for open-dialog conversational chatbots](https://github.com/ncoop57/i-am-a-nerd/blob/master/_notebooks/2020-05-12-chatbot-part-1.ipynb) by [Nathan Cooper](https://github.com/ncoop57)
- [Make your own Rick Sanchez (bot) with Transformers and DialoGPT fine-tuning](https://colab.research.google.com/drive/15wa925dj7jvdvrz8_z3vU7btqAFQLVlG) by [Rostyslav Neskorozhenyi](https://www.linkedin.com/in/slanj)

## Setup

```shell
pip install -e .
```

## Usage

### Trainer

```shell
solipsis-trainer --output_dir=../models/ERICA \
    --data_filename=../data/empathetic_dialogue_processed_train--cleaned128.csv \
    --filter_by="speaker==<s1>" \
    --model_name_or_path="microsoft/DialoGPT-medium" \
    --config_name="microsoft/DialoGPT-medium" \
    --tokenizer_name="microsoft/DialoGPT-medium" \
    --text_key="text" \
    --num_history=7
```

OR

```shell
solipsis-trainer --output_dir=../models/ERICA-2024-05-24 \
    --data_filename=../data/ed_train__train-cleaned128.csv \
    --model_name_or_path="microsoft/DialoGPT-medium" \
    --config_name="microsoft/DialoGPT-medium" \
    --tokenizer_name="microsoft/DialoGPT-medium" \
    --text_key="response" \
    --num_history=7
```

### Export

```shell
solipsis-exporter -m ../models/ERICA -t ../models/ERICA -o ../models/ERICA--exported.tar.gz
```

OR

```shell
solipsis-exporter -m ../models/ERICA-2024-05-24 -t ../models/ERICA/ERICA-2024-05-24 -o ../models/ERICA-2024-05-24--exported.tar.gz
```

### Chat

```shell
solipsis-chat -m ../models/ERICA -t ../models/ERICA -c ../models/ERICA -cf ../models/distilroberta-finetuned -d "cpu"
```

OR

```shell
solipsis-chat -m ../models/ERICA-2024-05-24 -t ../models/ERICA-2024-05-24 -c ../models/ERICA-2024-05-24 -cf ../models/distilroberta-finetuned -d "cpu"
```