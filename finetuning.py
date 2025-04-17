# finetuning.py
# Launch with: python -m torch.distributed.launch --nproc_per_node=8 finetuning.py

import os
import random
import types
from os.path import join, dirname
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import esm
from datasets import Dataset
import polars as pl

from transformers import TrainingArguments, Trainer
from transformers.modeling_outputs import MaskedLMOutput

from peft import LoraConfig, get_peft_model, TaskType

from dotenv import load_dotenv
import wandb

# 0. Setup W&B
dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
wandb.login()
os.environ["WANDB_PROJECT"] = "esm2-protein-finetuning"

# 1. Prepare Dataset
from exploratory_data_analysis import proteome_eda_df

protein_counts = (
    proteome_eda_df
    .group_by("protein")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
print("\nUnique protein types:")
with pl.Config() as cfg:
    cfg.set_tbl_cols(protein_counts.width)
    cfg.set_tbl_rows(protein_counts.height)
    print(protein_counts)

finetune_dataset = Dataset.from_polars(proteome_eda_df)
to_remove = list(set(finetune_dataset.column_names) - {"sequence"})
finetune_dataset = finetune_dataset.remove_columns(to_remove)

splits    = finetune_dataset.train_test_split(test_size=0.2, seed=42)
train_raw = splits["train"]
tmp       = splits["test"].train_test_split(test_size=0.5, seed=42)
eval_raw  = tmp["train"]
test_raw  = tmp["test"]

# 2. Load FAIR‑ESM model & tokenizer helper
esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter     = alphabet.get_batch_converter()
mask_token_id       = alphabet.mask_idx
vocab_size          = len(alphabet.all_toks)      # fixed vocab-size lookup

def tokenize_function(examples):
    data = [(str(i), seq) for i, seq in enumerate(examples["sequence"])]
    _, _, tokens = batch_converter(data)
    return {"input_ids": tokens}

train_dataset = train_raw.map(tokenize_function, batched=True, remove_columns=["sequence"])
eval_dataset  = eval_raw.map(tokenize_function, batched=True, remove_columns=["sequence"])
test_dataset  = test_raw.map(tokenize_function, batched=True, remove_columns=["sequence"])

# 3. Wrap ESM‑2 in a MaskedLM interface (now tolerates attention_mask)
class ESMForMaskedLM(torch.nn.Module):
    def __init__(self, esm_model, hidden_size, vocab_size):
        super().__init__()
        self.esm = esm_model
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # ignore attention_mask/kwargs, ESM only needs input_ids
        esm_out = self.esm(input_ids, repr_layers=[], return_contacts=False)
        hidden = esm_out["logits"]                   # (batch, seq_len, hidden_size)
        logits = self.lm_head(hidden)                # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return MaskedLMOutput(loss=loss, logits=logits)

wrapped_model = ESMForMaskedLM(
    esm_model=esm_model,
    hidden_size=esm_model.embed_dim,
    vocab_size=vocab_size,
)

wrapped_model.gradient_checkpointing_enable()

# 4. Patch for PEFT
config_dict = {
    "hidden_size":       esm_model.embed_dim,
    "num_hidden_layers": esm_model.num_layers,
}
wrapped_model.config = SimpleNamespace(**config_dict)
wrapped_model.config.to_dict = lambda: config_dict

def _passthrough_prepare(self, input_ids, **kwargs):
    return {"input_ids": input_ids, **kwargs}

wrapped_model.prepare_inputs_for_generation = types.MethodType(
    _passthrough_prepare, wrapped_model
)

# 5. Apply PEFT‑LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ],
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(wrapped_model, lora_config)
model.print_trainable_parameters()

# 6. Data collator with padding + MLM masking
def mask_tokens(batch: torch.Tensor, mlm_prob=0.15):
    labels = batch.clone()
    mask   = torch.zeros_like(batch, dtype=torch.bool)
    seq_len = batch.size(1)
    for i in range(batch.size(0)):
        n = max(1, int(seq_len * mlm_prob))
        idx = random.sample(range(seq_len), n)
        mask[i, idx] = True

    inputs_masked = batch.masked_fill(mask, mask_token_id)
    labels[~mask] = -100
    return {"input_ids": inputs_masked, "labels": labels}

def data_collator(features):
    seqs = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    batch = pad_sequence(seqs, batch_first=True, padding_value=alphabet.padding_idx)
    return mask_tokens(batch)

# 7. TrainingArguments & Trainer
training_args = TrainingArguments(
    output_dir="./v1-esm2-spider-silk-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2, 
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    save_total_limit=4,
    fp16=True,
    report_to=["wandb"],
    ddp_find_unused_parameters=False,
    run_name="v1-ESM2_FineTune_SpiderSilk",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# 8. Launch training
if __name__ == "__main__":
    trainer.train()
