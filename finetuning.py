from datasets import Dataset
from exploratory_data_analysis import proteome_eda_df
import polars as pl

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

import os
from os.path import join, dirname
from dotenv import load_dotenv
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

import wandb
wandb.login()
os.environ["WANDB_PROJECT"] = "esm2-protein-finetuning"

# Unique protein types
protein_counts = proteome_eda_df.group_by("protein").agg(pl.len().alias("count")).sort("count", descending=True)
print("\nUnique protein types:") # Show everything
with pl.Config() as cfg:
    cfg.set_tbl_cols(protein_counts.width)  # Set number of columns to display
    cfg.set_tbl_rows(protein_counts.height)  # Set number of rows to display
    print(protein_counts)

# For first-finetuning use full dataset, than could cut-down protein types to see if embeddings improve
"""
Protein types with physical traits (can be used for final evaluation):
（物性データがある->すなわち、identifyしたmotifと物性的特徴に相関関係があるかをある程度定量的に測ることができる。）
    - MiSp
    - MaSp1
    - MaSp
    - MaSp2
    - MaSp3b
    - MaSp3
    - MaSp2b
    - Ampullate spidroin
"""
# ---------------------------
# 1. Load Polars DataFrame
# ---------------------------
proteome_eda_df.head()

finetune_dataset = Dataset.from_polars(proteome_eda_df)
columns_to_remove = list(set(finetune_dataset.column_names) - {"sequence"})
finetune_dataset = finetune_dataset.remove_columns(columns_to_remove)

split_dataset = finetune_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset_raw = split_dataset["train"]
eval_dataset_raw = split_dataset["test"]

eval_split = eval_dataset_raw.train_test_split(test_size=0.5, seed=42) # 50% of eval set will become test
eval_dataset = eval_split["train"]
test_dataset = eval_split["test"]

# ---------------------------
# 2. Load the ESM-2 Tokenizer and Model
# ---------------------------
# Select a checkpoint (adjust based on GPU memory, e.g., "facebook/esm2_t30_150M_UR50D")
model_name = "facebook/esm2_t30_150M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Display special tokens to verify proper tokenizer loading
print("Mask token:", tokenizer.mask_token, "ID:", tokenizer.mask_token_id)
print("Pad token:", tokenizer.pad_token, "ID:", tokenizer.pad_token_id)
print("Vocabulary size:", tokenizer.vocab_size)

# ---------------------------
# 3. Tokenize the Protein Sequences
# ---------------------------
def tokenize_function(examples):
    # Tokenize sequences; 'truncation=True' ensures sequences longer than model_max_length are truncated.
    return tokenizer(examples["sequence"], truncation=True)

# Tokenize the entire dataset. Using batched processing for speed.
train_dataset = train_dataset_raw.map(tokenize_function, batched=True, remove_columns=["sequence"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["sequence"])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["sequence"])

# ---------------------------
# 4. Apply PEFT-LoRA on the Base Model
# ---------------------------
# Configure LoRA. Adjust parameters (r, lora_alpha, lora_dropout) as needed.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  
    inference_mode=False,         # set True only when in inference
    target_modules=["query", "value"],
    r=8,                          # LoRA rank (adjust based on desired adaptation capacity)
    lora_alpha=32,                # scaling factor
    lora_dropout=0.1              # dropout applied on LoRA layers
)

# Wrap the base model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # (Optional) Prints which parameters will be trained

# ---------------------------
# 4. Create the Data Collator for MLM
# ---------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # You can adjust this (usually 15%)
)

training_args = TrainingArguments(
    output_dir="./v1-esm2-spider-silk-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,                   # Adjust based on your training regime
    per_device_train_batch_size=8,        # Adjust according to your TPU/GPU memory
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,                    # Log every 100 steps
    save_steps=100,                       # Save checkpoints every 500 steps
    save_total_limit=4,
    fp16=True,                           # Disable fp16 (set to True if you are on GPUs and want mixed precision)
    bf16=False,                            # Enable bf16 mixed precision (commonly used on TPUs)
    report_to=["wandb"],                  # Enable wandb logging
    ddp_find_unused_parameters=False,
    run_name="v1-ESM2_FineTune_SpiderSilk"  # Name for this wandb run
)

# ---------------------------
# 6. Initialize Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer  
)

# ---------------------------
# 7. Start the Fine-Tuning Process
# ---------------------------
if __name__ == "__main__":
    trainer.train()
