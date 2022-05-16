import torch, os, re, pandas as pd, json
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
from datasets import Dataset, list_metrics, load_metric


def prepare_data(
    data_or_filename: Union[str, pd.DataFrame],
    filter_by: str = None,
    filter_value: str = None,
    content_key: str = "text",
    n: int = 7,
    test_size: float = 0.1,
):
    data = load_csv(data_or_filename) if isinstance(data_or_filename, str) else data_or_filename

    contexted_data = prepare_context(
        data,
        filter_by=filter_by,
        filter_value=filter_value,
        content_key=content_key,
        n=n,
    )

    trn_df, val_df = train_test_split(contexted_data, test_size=test_size, shuffle=False)
    
    return trn_df, val_df


def prepare_context(
    data: pd.DataFrame,
    filter_by: str = None,
    filter_value: str = None,
    content_key: str = "text",
    n: int = 7,
):
    if filter_by:
        indexes = data.loc[data[filter_by] == filter_value].index
        for idx, i in enumerate(indexes):
            if i > n:
                break
        indexes = indexes[idx:]
    else:
        indexes = range(n, len(data[content_key]))
        
    contexted = []

    for i in indexes:
        row = []
        prev = i - 1 - n
        for j in range(i, prev, -1):
            row.append(data.iloc[j][content_key])
        contexted.append(row)
            
    columns = ["response", "context"]
    columns = columns + ["context/" + str(i) for i in range(n - 1)]

    print(columns)
    df = pd.DataFrame.from_records(contexted, columns=columns)
    
    return df


def preprocess_function(tokenizer, max_length=512):
    def _construct(examples):
        flatten = lambda l: [item for sublist in l for item in sublist] 
        concat_text = f"{tokenizer.eos_token}".join(reversed([v for _, v in examples.items() if isinstance(v, str)]))
        concat_text = concat_text + tokenizer.eos_token
        tokenized = tokenizer(concat_text, padding="max_length",  max_length=max_length)
        examples["input_ids"] = tokenized["input_ids"]
        examples["attention_mask"] = tokenized["attention_mask"]
        return examples
        
    return _construct


def compute_metrics(eval_pred):
    metric = load_metric("perplexity")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


## Chat App
def generate_responses(model, tokenizer, text, chat_history_ids=None, step=0):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=512,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7
    )
    
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True,
    )

    return response, chat_history_ids, step + 1


def chat(model, tokenizer):
    step = 0
    chat_history_ids = []
    
    while True: 
        text = input(">> ")
        if text in ["/q", "/quit", "/e", "/exit"]: break
        print(f"User: {text}")
        response, chat_history_ids, step = generate_responses(
            model=model,
            tokenizer=tokenizer,
            text=text,
            chat_history_ids=chat_history_ids,
            step=step
        )
        print(f"Bot: {response}")


def train():
    MODEL_NAME = "microsoft/DialoGPT-small"
    model_cls = AutoModelForCausalLM
    tokenizer_cls = AutoTokenizer
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  

    device = torch.device(dev) 

    print(f"""
        cuda: {torch.cuda.is_available()}
        current_device: {torch.cuda.current_device()}
        device_count: {torch.cuda.device_count()}
    """)
    
    # Load Config
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    # Load data
    filepath = "../data/processed.csv"
    df = pd.read_csv(filepath, encoding="utf-8", usecols=["character", "content"]).rename(columns={"content": "text"})
    
    # Setup tokenizer
    base_tokenizer = tokenizer_cls.from_pretrained(MODEL_NAME)
    base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Setup base model
    base_model = model_cls.from_pretrained(MODEL_NAME, config=config)
    base_model.resize_token_embeddings(len(base_tokenizer))
    
    # Preprocess data
    train_df, val_df = prepare_data(df, filter_by="character", filter_value="bitjockey")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    tokenized_train_dataset = train_dataset.map(preprocess_function(tokenizer=base_tokenizer, max_length=256), remove_columns=list(train_dataset.features.keys()))
    tokenized_val_dataset = val_dataset.map(preprocess_function(tokenizer=base_tokenizer, max_length=256), remove_columns=list(val_dataset.features.keys()))
    
    # Convert to tensors
    tokenized_train_dataset.set_format(type="torch", columns=["input_ids"])
    tokenized_val_dataset.set_format(type="torch", columns=["input_ids"])
    FINETUNED_MODEL = 'CHARLOTTE-05162022a-myDialoGPT2-small'

    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    # Setup trainer
    trainer = None
    training_args = None
    training_args = TrainingArguments(
        output_dir=FINETUNED_MODEL,          # output directory
        evaluation_strategy="epoch",
        num_train_epochs=3,           # total # of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,   # batch size for evaluation
        weight_decay=0.01,           # strength of weight decay
        logging_dir=FINETUNED_MODEL,            # directory for storing logs
        prediction_loss_only=True,
    )
    
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    # Run training 
    trainer.train()
    
    print("Training completed")
    
    # Save model, tokenizer, and config
    trainer.save_model(FINETUNED_MODEL)
    base_tokenizer.save_pretrained(FINETUNED_MODEL)
    config.save_pretrained(FINETUNED_MODEL)
    
    print("Run the code below in a Jupyter cell") 
    print(
        f"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        finetuned_model = AutoModelForCausalLM.from_pretrained({FINETUNED_MODEL})
        tokenizer = AutoTokenizer.from_pretrained({FINETUNED_MODEL})
        
        # Start the chat
        chat(finetuned_model, tokenizer)
        """
    )