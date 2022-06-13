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
export TIMESTAMP=$(date +"%Y-%m-%d-%H:%M:%S")
export OUTPUT_DIR=ERICA-medium__$TIMESTAMP
export DATA_FILENAME=../data/empathetic_dialogue_processed.csv
export MODEL_BASE="microsoft/DialoGPT-medium"
export FILTER_BY="speaker==<s1>"

solipsis-trainer --output_dir $OUTPUT_DIR \
    --data_filename $DATA_FILENAME \
    --filter_by $FILTER_BY \
    --model_name_or_path=$MODEL_BASE \
    --config_name=$MODEL_BASE \
    --tokenizer_name=$MODEL_BASE \
    --content_key="text"


export TIMESTAMP=$(date +"%Y-%m-%d-%H:%M:%S")
export OUTPUT_DIR=hopperbot-medium__$TIMESTAMP
export DATA_FILENAME=../data/processed.csv
export MODEL_BASE="microsoft/DialoGPT-medium"
export FILTER_BY="character==bitjockey"
```