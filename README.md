# SolipsisAI - research

This is a library used for training the models in our apps.

- [SolipsisAI - research](#solipsisai---research)
  - [Credits](#credits)
  - [References](#references)
  - [Setup](#setup)
  - [Usage](#usage)

## Credits

The scripts/code here were heavily lifted from:

- [How to fine-tune the DialoGPT model on a new dataset for open-dialog conversational chatbots](https://github.com/ncoop57/i-am-a-nerd/blob/master/_notebooks/2020-05-12-chatbot-part-1.ipynb) by [Nathan Cooper](https://github.com/ncoop57)
- [Make your own Rick Sanchez (bot) with Transformers and DialoGPT fine-tuning](https://colab.research.google.com/drive/15wa925dj7jvdvrz8_z3vU7btqAFQLVlG) by [Rostyslav Neskorozhenyi](https://www.linkedin.com/in/slanj)

## References

- [Beginner’s Guide To Building A Singlish AI Chatbot](https://towardsdatascience.com/beginners-guide-to-building-a-singlish-ai-chatbot-7ecff8255ee)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)

## Setup

```shell
pip install -e .
```

## Usage

```shell
solipsis-trainer --output_dir ../models/hopperbot-medium \
    --data_filename ../data/processed.csv \
    --filter_by "character==bitjockey" \
    --model_name_or_path="microsoft/DialoGPT-medium" \
    --config_name="microsoft/DialoGPT-medium" \
    --tokenizer_name="microsoft/DialoGPT-medium" \
    --evaluate_during_training 

solipsis-trainer --output_dir ../models/charlottebot-medium \
    --data_filename ../data/processed.csv \
    --filter_by "character==Charlotte" \
    --model_name_or_path "microsoft/DialoGPT-medium" \
    --config_name "microsoft/DialoGPT-medium" \
    --tokenizer_name "microsoft/DialoGPT-medium" \
    --per_gpu_eval_batch_size 2 \
    --per_gpu_train_batch_size 2 \
    --evaluate_during_training
```