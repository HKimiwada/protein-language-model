# Evaluation was run on google colab
import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel
import esm
from types import SimpleNamespace
import types
from transformers.modeling_outputs import MaskedLMOutput

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
base_mlm = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t30_150M_UR50D")
base_mlm.eval()
_, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

from transformers import AutoTokenizer, AutoModelForMaskedLM
import esm
import torch
from types import SimpleNamespace
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput

# -----------------------------------------------------------------------------
# 0) Raw ESM backbone + alphabet (for wrapping + masking indices)
# -----------------------------------------------------------------------------
esm_backbone, alphabet = esm.pretrained.esm2_t30_150M_UR50D()

# -----------------------------------------------------------------------------
# 1) Your fine‑tuned wrapper
# -----------------------------------------------------------------------------
class ESMForMaskedLM(torch.nn.Module):
    def __init__(self, esm_model, vocab_size):
        super().__init__()
        self.esm         = esm_model
        self.hidden_size = esm_model.embed_dim
        self.lm_head     = torch.nn.Linear(self.hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        out = self.esm(
            input_ids,
            repr_layers=[self.esm.num_layers],
            return_contacts=False
        )
        hidden = out["representations"][self.esm.num_layers]
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return MaskedLMOutput(loss=loss, logits=logits)

# -----------------------------------------------------------------------------
# 2) Instantiate & load your checkpoint into that wrapper
# -----------------------------------------------------------------------------
vocab_size = len(alphabet.all_toks)
ft_model   = ESMForMaskedLM(esm_model=esm_backbone, vocab_size=vocab_size)

# dummy config for HF compatibility
cfg = {"hidden_size": ft_model.hidden_size, "num_hidden_layers": ft_model.esm.num_layers}
ft_model.config = SimpleNamespace(**cfg)
ft_model.config.to_dict = lambda: cfg

# load your fine‑tuned weights
state_dict = torch.load(
    "/content/drive/MyDrive/研究関連/大学/03_Silkome_EDA_Finetuning/v1-esm2-spider-silk-finetuned/checkpoint-6600/pytorch_model.bin",
    map_location="cpu"
)
state = {k: v for k, v in state_dict.items() if k in ft_model.state_dict()}
ft_model.load_state_dict(state, strict=False)
ft_model.eval()

# =============================================================================
# 1. Data collator (80/10/10 masking)
# =============================================================================
class DataCollatorForProteinMLM:
    def __init__(self, alphabet, mlm_prob=0.15):
        self.alphabet = alphabet
        self.mlm_prob = mlm_prob
        self.vocab_size = len(alphabet.all_toks)
    def __call__(self, tokens: torch.LongTensor):
        # tokens: (B, L)
        prob_mat = torch.full(tokens.shape, self.mlm_prob, device=tokens.device)
        prob_mat.masked_fill_(tokens.eq(self.alphabet.padding_idx), 0.0)

        masked_indices = torch.bernoulli(prob_mat).bool()
        labels         = tokens.clone()
        labels[~masked_indices] = -100

        rand      = torch.rand(tokens.shape, device=tokens.device)
        mask_inds = masked_indices & (rand < 0.8)
        rand_inds = masked_indices & (rand >= 0.8) & (rand < 0.9)

        masked_tokens = tokens.clone()
        masked_tokens[mask_inds] = self.alphabet.mask_idx

        random_tokens = torch.randint(1, self.vocab_size, tokens.shape, device=tokens.device)
        masked_tokens[rand_inds] = random_tokens[rand_inds]

        return masked_tokens, labels, masked_indices

# =============================================================================
# 2. DataLoader wrapping your HF Dataset
# =============================================================================
from datasets import load_from_disk
test_ds = load_from_disk("/content/drive/MyDrive/研究関連/大学/03_Silkome_EDA_Finetuning/v2-esm2-spider-silk-finetuned/tokenized_esm2_v2_test")

def collate_fn(batch):
    ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    mask = (ids != alphabet.padding_idx).long()
    return {"input_ids": ids, "attention_mask": mask}

bpr_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
acc_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

# =============================================================================
# 4. Metrics on raw IDs
# =============================================================================
def calculate_bits_per_residue_from_ids(model, dataloader, alphabet):
    device         = next(model.parameters()).device
    collator       = DataCollatorForProteinMLM(alphabet)
    total_nats     = 0.0
    total_masked   = 0
    model.eval()

    for batch in tqdm(dataloader, desc="BPR Eval"):
        tokens         = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        masked_tokens, labels, _ = collator(tokens)
        with torch.no_grad():
            out    = model(masked_tokens, attention_mask=attention_mask)
            logits = out.logits

        log_probs = F.log_softmax(logits, dim=-1)
        nll       = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            labels.view(-1),
            reduction="sum",
            ignore_index=-100,
        )

        total_nats   += nll.item()
        total_masked += (labels != -100).sum().item()

    return total_nats / (total_masked * math.log(2))


def bucket_accuracy_from_ids(model, dataloader, alphabet, buckets=[100,300]):
    device   = next(model.parameters()).device
    collator = DataCollatorForProteinMLM(alphabet)
    results  = {}
    model.eval()

    bucket_defs = {
        "<=100":   (0, buckets[0]),
        "101-300": (buckets[0]+1, buckets[1]),
        ">300":    (buckets[1]+1, float("inf")),
    }

    for name, (low, high) in bucket_defs.items():
        correct, total = 0, 0
        for batch in dataloader:
            tokens         = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths        = (tokens != alphabet.padding_idx).sum(dim=1)
            mask_batch     = (lengths >= low) & (lengths <= high)
            if not mask_batch.any():
                continue

            masked_tokens, labels, _ = collator(tokens)
            with torch.no_grad():
                preds = model(masked_tokens, attention_mask=attention_mask).logits.argmax(dim=-1)

            for i in torch.where(mask_batch)[0].tolist():
                m = labels[i] != -100
                correct += (preds[i][m] == tokens[i][m]).sum().item()
                total   +=    m.sum().item()

        results[name] = correct / total if total > 0 else None

    return results


def calculate_per_sequence_bpr_from_ids(model, dataloader, alphabet):
    # single‐sequence loader
    single_loader = DataLoader(dataloader.dataset, batch_size=1,
                               shuffle=False, collate_fn=collate_fn)
    vals = []
    for batch in tqdm(single_loader, desc="Per‑seq BPR"):
        bpr = calculate_bits_per_residue_from_ids(model, single_loader, alphabet)
        vals.append(bpr)

    import numpy as np
    arr = np.array(vals)
    stats = {
        "mean":   arr.mean(),
        "median": np.median(arr),
        "std":    arr.std(),
        "q1":     np.percentile(arr, 25),
        "q3":     np.percentile(arr, 75),
        "min":    arr.min(),
        "max":    arr.max(),
    }
    print("Per‑sequence BPR stats:", stats)
    return vals, stats


def calculate_per_aa_nll_from_ids(model, dataloader, alphabet, trials=5):
    device   = next(model.parameters()).device
    collator = DataCollatorForProteinMLM(alphabet)
    aa_nll   = {}
    aa_cnt   = {}
    aa_cor   = {}
    model.eval()

    single_loader = DataLoader(dataloader.dataset, batch_size=1,
                               shuffle=False, collate_fn=collate_fn)
    for idx, batch in enumerate(single_loader):
        if idx >= 100:
            break
        tokens         = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        for _ in range(trials):
            masked_tokens, labels, masked_idx = collator(tokens)
            with torch.no_grad():
                out    = model(masked_tokens, attention_mask=attention_mask)
                logits = out.logits
                logp   = F.log_softmax(logits, dim=-1)

            for b, pos in masked_idx.nonzero(as_tuple=False):
                true_id = tokens[b, pos].item()
                aa      = alphabet.get_tok(true_id)

                aa_nll[aa] = aa_nll.get(aa, 0.0) - logp[b, pos, true_id].item()
                aa_cnt[aa] = aa_cnt.get(aa, 0) + 1
                aa_cor[aa] = aa_cor.get(aa, 0) + int(logits[b, pos].argmax() == true_id)

    return {
        aa: {
            "avg_nll": aa_nll[aa] / aa_cnt[aa],
            "accuracy": aa_cor[aa] / aa_cnt[aa],
        }
        for aa in aa_cnt
    }


# =============================================================================
# 5. Driver to compare base vs fine‑tuned
# =============================================================================
def evaluate_mlm_model_ids(base_model, ft_model, bpr_loader, acc_loader, alphabet):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)
    ft_model.to(device)

    summary = {
        "bpr": {
            "base": calculate_bits_per_residue_from_ids(base_model, bpr_loader, alphabet),
            "ft":   calculate_bits_per_residue_from_ids(ft_model,   bpr_loader, alphabet),
        },
        "per_seq_stats_base": calculate_per_sequence_bpr_from_ids(base_model, bpr_loader, alphabet)[1],
        "per_seq_stats_ft":   calculate_per_sequence_bpr_from_ids(ft_model,   bpr_loader, alphabet)[1],
        "bucket_acc": {
            "base": bucket_accuracy_from_ids(base_model, bpr_loader, alphabet),
            "ft":   bucket_accuracy_from_ids(ft_model,   bpr_loader, alphabet),
        },
        "aa_stats": {
            "base": calculate_per_aa_nll_from_ids(base_model, bpr_loader, alphabet),
            "ft":   calculate_per_aa_nll_from_ids(ft_model,   bpr_loader, alphabet),
        },
    }

    print("Evaluation summary:", summary)
    return summary


# =============================================================================
# 6. Run evaluation
# =============================================================================
summary = evaluate_mlm_model_ids(
    base_model = base_mlm,
    ft_model   = ft_model,
    bpr_loader = bpr_loader,
    acc_loader = acc_loader,
    alphabet   = alphabet
)