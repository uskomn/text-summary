from datasets import load_dataset
from transformers import BartTokenizer

def get_dataloaders(config):
    tokenizer = BartTokenizer.from_pretrained(config.model_name)

    def preprocess(batch):
        inputs = tokenizer(batch['article'], truncation=True, padding='max_length', max_length=config.max_input_length)
        targets = tokenizer(batch['highlights'], truncation=True, padding='max_length', max_length=config.max_target_length)
        inputs['labels'] = targets['input_ids']
        return inputs

    dataset = load_dataset("cnn_dailymail", "3.0.0",cache_dir=config.cache_dir)
    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset['train'].column_names)

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized
