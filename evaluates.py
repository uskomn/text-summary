import torch
import evaluate
from transformers import BartForConditionalGeneration, BartTokenizer
from config import Config
from dataset import get_dataloaders
from torch.utils.data import DataLoader

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartForConditionalGeneration.from_pretrained(config.model_name).to(device)
tokenizer = BartTokenizer.from_pretrained(config.model_name)

tokenized = get_dataloaders(config)
val_loader = DataLoader(tokenized['validation'], batch_size=config.eval_batch_size)

metric = evaluate.load("rouge")

model.eval()
for batch in val_loader:
    inputs = {k: torch.tensor(v).to(device) for k, v in batch.items()}
    with torch.no_grad():
        summaries = model.generate(input_ids=inputs['input_ids'], max_length=config.max_target_length, num_beams=4)

    decoded_preds = tokenizer.batch_decode(summaries, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)

    metric.add_batch(predictions=decoded_preds, references=decoded_labels)

results = metric.compute()
for key in ['rouge1', 'rouge2', 'rougeL']:
    print(f"{key}: {results[key].mid.fmeasure:.4f}")
