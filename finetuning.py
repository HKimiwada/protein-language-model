from datasets import Dataset
from exploratory_data_analysis import proteome_eda_df
import polars as pl

# HF Trainer & PEFT imports
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

import os
from os.path import join, dirname
from dotenv import load_dotenv
import wandb

# FAIR‑ESM import
import esm
import torch
import random

# 0. Setup W&B & env
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login()
os.environ["WANDB_PROJECT"] = "esm2-protein-finetuning"

# ---------------------------
# 1. Prepare Dataset
# ---------------------------
# Show unique protein types
protein_counts = proteome_eda_df.group_by("protein") \
    .agg(pl.len().alias("count")) \
    .sort("count", descending=True)
print("\nUnique protein types:")
with pl.Config() as cfg:
    cfg.set_tbl_cols(protein_counts.width)
    cfg.set_tbl_rows(protein_counts.height)
    print(protein_counts)

# Convert to Hugging Face Dataset
finetune_dataset = Dataset.from_polars(proteome_eda_df)
to_remove = list(set(finetune_dataset.column_names) - {"sequence"})
finetune_dataset = finetune_dataset.remove_columns(to_remove)

splits = finetune_dataset.train_test_split(test_size=0.2, seed=42)
train_raw = splits["train"]
eval_raw  = splits["test"].train_test_split(test_size=0.5, seed=42)
eval_set  = eval_raw["train"]
test_set  = eval_raw["test"]

# ---------------------------
# 2. Load FAIR‑ESM Model & Tokenizer
# ---------------------------
# 2.1 Load pretrained ESM‑2 model & alphabet
model, alphabet   = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter   = alphabet.get_batch_converter()
mask_token_id     = alphabet.mask_idx

# 2.2 Tokenization function
def tokenize_function(examples):
    # examples["sequence"] is the raw AA string
    data = [(str(i), seq) for i, seq in enumerate(examples["sequence"])]
    _, _, batch_tokens = batch_converter(data)
    return {"input_ids": batch_tokens}

# Apply tokenization
train_dataset = train_raw.map(
    tokenize_function, batched=True, remove_columns=["sequence"]
)
eval_dataset  = eval_set.map(
    tokenize_function, batched=True, remove_columns=["sequence"]
)
test_dataset  = test_set.map(
    tokenize_function, batched=True, remove_columns=["sequence"]
)

# ── ADD THIS PATCH FOR PEFT ───────────────────────────────────────────
from types import SimpleNamespace

# Build a minimal config dict with the attributes PEFT needs
config_dict = {
    "hidden_size":      model.embed_dim,       # embedding dimension
    "num_hidden_layers": model.num_layers      # number of transformer blocks
}

# Attach a config object with to_dict()
model.config = SimpleNamespace(**config_dict)
model.config.to_dict = lambda: config_dict
# ─────────────────────────────────────────────────────────────────────

# ---------------------------
# 3. PEFT‑LoRA Setup
# ---------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    target_modules=["query", "value"],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------
# 4. Data Collator for MLM
# ---------------------------
def mask_tokens(inputs, mlm_prob=0.15):
    """
    Simple MLM masking: 80% mask, 10% random, 10% original.
    """
    labels = inputs.clone()
    batch_size, seq_len = labels.shape
    # pick tokens to mask
    mask = torch.full(labels.shape, False, dtype=torch.bool)
    for i in range(batch_size):
        num_to_mask = max(1, int(seq_len * mlm_prob))
        idx = random.sample(range(seq_len), num_to_mask)
        mask[i, idx] = True

    inputs_masked = inputs.clone()
    inputs_masked[mask] = mask_token_id
    labels[~mask] = -100  # only compute loss on masked tokens
    return {"input_ids": inputs_masked, "labels": labels}

data_collator = lambda features: mask_tokens(
    torch.stack([f["input_ids"] for f in features])
)

# ---------------------------
# 5. Training Arguments & Trainer
# ---------------------------
training_args = TrainingArguments(
    output_dir="./v1-esm2-spider-silk-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    save_total_limit=4,
    fp16=True,
    report_to=["wandb"],
    ddp_find_unused_parameters=False,
    run_name="v1-ESM2_FineTune_SpiderSilk"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# ---------------------------
# 6. Start Fine‑Tuning
# ---------------------------
if __name__ == "__main__":
    trainer.train()
