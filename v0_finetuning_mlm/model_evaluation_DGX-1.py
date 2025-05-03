# Model evaluation code for DGX-1 (for CUDA 11.0)
# evaluation.py
import os
import argparse
import math
import json
import urllib.request

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence  # :contentReference[oaicite:0]{index=0}
from tqdm import tqdm

import esm                                        # :contentReference[oaicite:1]{index=1}
from transformers.modeling_outputs import MaskedLMOutput
from datasets import load_from_disk

# -----------------------------------------------------------------------------
# 1. Fetch ESM-2 150M masked-LM checkpoint (backbone + head)
# -----------------------------------------------------------------------------
ESM2_PT = "esm2_t30_150M_UR50D.pt"
if not os.path.exists(ESM2_PT):
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",  # 
        ESM2_PT
    )

# -----------------------------------------------------------------------------
# 2. Load backbone & Alphabet (vocab, mask/pad indices)
# -----------------------------------------------------------------------------
esm_backbone, alphabet = esm.pretrained.esm2_t30_150M_UR50D()  # :contentReference[oaicite:2]{index=2}
VOCAB_SIZE             = len(alphabet.all_toks)

# -----------------------------------------------------------------------------
# 3. Fine‑tuned wrapper for masked‑LM
# -----------------------------------------------------------------------------
class ESMForMaskedLM(torch.nn.Module):
    def __init__(self, esm_model, vocab_size):
        super().__init__()
        self.esm         = esm_model
        self.hidden_size = esm_model.embed_dim
        self.lm_head     = torch.nn.Linear(self.hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
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
# 4. MLM masking collator (80/10/10)
# -----------------------------------------------------------------------------
class DataCollatorForProteinMLM:
    def __init__(self, alphabet, mlm_prob=0.15):
        self.alphabet = alphabet
        self.mlm_prob = mlm_prob
        self.vocab_size = len(alphabet.all_toks)

    def __call__(self, tokens: torch.LongTensor):
        prob = torch.full(tokens.shape, self.mlm_prob, device=tokens.device)
        prob.masked_fill_(tokens.eq(self.alphabet.padding_idx), 0.0)
        masked = torch.bernoulli(prob).bool()

        labels = tokens.clone()
        labels[~masked] = -100

        rand = torch.rand(tokens.shape, device=tokens.device)
        mask_inds = masked & (rand < 0.8)
        rand_inds = masked & (rand >= 0.8) & (rand < 0.9)

        corrupted = tokens.clone()
        corrupted[mask_inds] = self.alphabet.mask_idx
        corrupted[rand_inds] = torch.randint(1, self.vocab_size, tokens.shape, device=tokens.device)[rand_inds]

        return corrupted, labels, masked

# -----------------------------------------------------------------------------
# 5. Batch padding & masking for pre‑tokenized input_ids
# -----------------------------------------------------------------------------
def collate_fn(batch):
    """
    batch: list of dicts, each with 'input_ids': List[int]
    returns: dict with 'input_ids' (LongTensor B×L_max) and 'attention_mask'
    """
    seqs = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    padded = pad_sequence(seqs, batch_first=True, padding_value=alphabet.padding_idx)  # :contentReference[oaicite:3]{index=3}
    attn   = (padded != alphabet.padding_idx).long()
    return {"input_ids": padded, "attention_mask": attn}

# -----------------------------------------------------------------------------
# 6. Evaluation metrics
# -----------------------------------------------------------------------------
def calculate_bits_per_residue(model, loader, device):
    collator = DataCollatorForProteinMLM(alphabet)
    total_nats = total_mask = 0
    model.eval()

    for batch in tqdm(loader, desc="BPR"):
        tokens = batch["input_ids"].to(device)
        attn   = batch["attention_mask"].to(device)
        corrupted, labels, _ = collator(tokens)

        with torch.no_grad():
            out = model(corrupted, attention_mask=attn)

        log_probs = F.log_softmax(out.logits, dim=-1)
        nll = F.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            labels.view(-1),
            reduction="sum", ignore_index=-100
        )
        total_nats += nll.item()
        total_mask += (labels != -100).sum().item()

    return total_nats / (total_mask * math.log(2))


def bucket_accuracy(model, loader, device, buckets=[100,300]):
    collator = DataCollatorForProteinMLM(alphabet)
    model.eval()
    results = {}

    defs = {"<=100": (0,buckets[0]), "101-300": (buckets[0]+1,buckets[1]), ">300": (buckets[1]+1,float("inf"))}
    for name,(lo,hi) in defs.items():
        correct = total = 0
        for batch in loader:
            tokens = batch["input_ids"].to(device)
            attn   = batch["attention_mask"].to(device)
            lengths = (tokens != alphabet.padding_idx).sum(dim=1)
            mask_seq = (lengths>=lo)&(lengths<=hi)
            if not mask_seq.any(): continue

            corrupted, labels, _ = collator(tokens)
            with torch.no_grad():
                preds = model(corrupted, attention_mask=attn).logits.argmax(dim=-1)

            for i in mask_seq.nonzero(as_tuple=False).squeeze(1).tolist():
                m = labels[i]!=-100
                correct += (preds[i][m]==tokens[i][m]).sum().item()
                total   += m.sum().item()

        results[name] = correct/total if total>0 else None
    return results

# -----------------------------------------------------------------------------
# 7. Main: DDP init, model loading, DataLoader, run eval
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Your fine‑tuned pytorch_model.bin")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="HF‐datasets folder with only 'input_ids'")
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--output",       type=str, default="summary.json")
    parser.add_argument("--local_rank", "--local-rank",
                        type=int, default=int(os.environ.get("LOCAL_RANK",0)),
                        help="DDP: process-local GPU index")
    args = parser.parse_args()

    # a) Initialize DDP (once) :contentReference[oaicite:4]{index=4}
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")

    # b) Base model: wrap backbone + head 
    base_wrapper = ESMForMaskedLM(esm_backbone, VOCAB_SIZE)
    base_state   = torch.load(ESM2_PT, map_location="cpu")
    base_wrapper.load_state_dict(
        {k:v for k,v in base_state.items() if k in base_wrapper.state_dict()},
        strict=False
    )
    base_ddp = DDP(base_wrapper.to(device).eval(),
                   device_ids=[args.local_rank])

    # c) Fine‑tuned model :contentReference[oaicite:5]{index=5}
    ft_backbone, _ = esm.pretrained.esm2_t30_150M_UR50D()
    ft_wrapper = ESMForMaskedLM(ft_backbone, VOCAB_SIZE)
    ft_state   = torch.load(args.checkpoint, map_location="cpu")
    ft_wrapper.load_state_dict(
        {k:v for k,v in ft_state.items() if k in ft_wrapper.state_dict()},
        strict=False
    )
    ft_ddp = DDP(ft_wrapper.to(device).eval(),
                 device_ids=[args.local_rank])

    # d) DataLoader :contentReference[oaicite:6]{index=6}
    ds      = load_from_disk(args.dataset_path)
    sampler = DistributedSampler(ds, shuffle=False)
    loader  = DataLoader(ds,
                         batch_size=args.batch_size,
                         sampler=sampler,
                         num_workers=args.workers,
                         pin_memory=True,
                         collate_fn=collate_fn)

    # e) Run evaluation
    bpr_base = calculate_bits_per_residue(base_ddp, loader, device)
    bpr_ft   = calculate_bits_per_residue(ft_ddp,   loader, device)
    acc_base = bucket_accuracy(base_ddp, loader, device)
    acc_ft   = bucket_accuracy(ft_ddp,   loader, device)

    summary = {"bpr": {"base": bpr_base, "ft": bpr_ft},
               "bucket_acc": {"base": acc_base, "ft": acc_ft}}

    # f) Write only on rank 0
    if dist.get_rank() == 0:
        with open(args.output, "w") as w:
            json.dump(summary, w, indent=2)
        print(f".. summary written to {args.output}")

if __name__ == "__main__":
    main()
