import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, AdamW, get_scheduler
from transformers import BartTokenizer
from config import Config
from dataset import get_dataloaders
from tqdm import tqdm
import os
import torch.nn.functional as F

# 配置与设备
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型与 tokenizer
model = BartForConditionalGeneration.from_pretrained(config.model_name,cache_dir=config.cache_dir).to(device)
tokenizer = BartTokenizer.from_pretrained(config.model_name,cache_dir=config.cache_dir)

# 加载数据
tokenized = get_dataloaders(config)
train_loader = DataLoader(tokenized['train'], batch_size=config.train_batch_size, shuffle=True)

# 优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=500,
                             num_training_steps=len(train_loader) * config.num_train_epochs)

# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

# 标签平滑函数
def label_smoothed_nll_loss(logits, target, epsilon, ignore_index=-100):
    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -log_probs.mean(dim=-1)

    pad_mask = target.eq(ignore_index)
    nll_loss.masked_fill_(pad_mask, 0.0)
    smooth_loss.masked_fill_(pad_mask, 0.0)

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / logits.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss / target.size(0)  # normalize by batch size

# 训练
model.train()
global_step = 0

for epoch in range(config.num_train_epochs):
    loop = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(loop):
        inputs = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=inputs['labels'])
            logits = outputs.logits
            loss = label_smoothed_nll_loss(logits, inputs['labels'], epsilon=config.label_smoothing)

        # 梯度累积
        scaler.scale(loss / config.gradient_accumulation_steps).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
            global_step += 1

        loop.set_postfix(loss=loss.item())

# 保存模型
os.makedirs(config.output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(config.output_dir, f"bart.pt"))
print("model saved")
