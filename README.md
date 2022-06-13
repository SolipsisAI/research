# SolipsisAI - research

This is a library used for training the models in our apps.

- [SolipsisAI - research](#solipsisai---research)
  - [Credits](#credits)
  - [References](#references)
  - [Setup](#setup)
  - [Usage](#usage)
    - [Training](#training)
    - [Chatting](#chatting)

## Credits

The scripts/code here were heavily lifted from:

- [🦄 How to build a State-of-the-Art Conversational AI with Transfer Learning](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313) by [Thom Wolf](http://thomwolf.io/)
- [How to fine-tune the DialoGPT model on a new dataset for open-dialog conversational chatbots](https://github.com/ncoop57/i-am-a-nerd/blob/master/_notebooks/2020-05-12-chatbot-part-1.ipynb) by [Nathan Cooper](https://github.com/ncoop57)
- [Make your own Rick Sanchez (bot) with Transformers and DialoGPT fine-tuning](https://colab.research.google.com/drive/15wa925dj7jvdvrz8_z3vU7btqAFQLVlG) by [Rostyslav Neskorozhenyi](https://www.linkedin.com/in/slanj)

## References

- [Beginner’s Guide To Building A Singlish AI Chatbot](https://towardsdatascience.com/beginners-guide-to-building-a-singlish-ai-chatbot-7ecff8255ee)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)

## Setup

```shell
git clone git@github.com:SolipsisAI/research.git solipsis-research
cd solipsis-research
pip install -e .
```

## Usage

### Training

```shell
solipsis-trainer --output_dir="../models/ERICA-medium" \
    --data_filename="../data/empathetic_dialogue_processed.csv" \
    --model_name_or_path="microsoft/DialoGPT-medium" \
    --config_name="microsoft/DialoGPT-medium" \
    --tokenizer_name="microsoft/DialoGPT-medium" \
    --evaluate_during_training 
```

### Chatting

```shell
solipsis-chat -m ../models/hopperbot-medium -t ../models/hopperbot-medum
```