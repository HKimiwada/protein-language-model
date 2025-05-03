# Storage Repository for Finetuning Protein Language Model for motif discovery
# Spider Silk Motif Discovery & Validation

A fully automated, GPU-accelerated pipeline to identify sequence motifs in spider silk proteins and link them statistically to fiber mechanics.

## 1. Inputs

- **Protein Sequences** (`data/fasta/spider-silkome-database.v1.prot.fasta`)  
- **Validation Dataset** (`data/validation/validation_dataset.csv` with `idv_id` + mechanical properties)  
- **Fine-tuned ESM-2 Model Checkpoint** (for embedding generation)

## 2. Pipeline Stages

1. **Residue Embedding**  
   Run fine-tuned ESM-2 on each sequence to produce per-residue vectors (DGX-1, 8× V100).

2. **Clustering**  
   Cluster residue embeddings (e.g. KMeans/DBSCAN) to find groups of similar local contexts.

3. **Window Extraction**  
   For each cluster, extract 15-mer sequence windows around member residues into `cluster_<n>.fasta`.

4. **Motif Discovery**  
   Invoke MEME Suite’s **STREME** on each cluster FASTA to identify enriched motifs; export consensus strings.

5. **Motif Validation**  
   - Merge validation CSV with sequences by `idv_id`.  
   - Count each motif’s (overlapping) occurrences per sequence.  
   - Compute Pearson & Spearman correlations with mechanical properties.  
   - Apply Benjamini–Hochberg FDR correction.  
   - (Optional) Generate scatterplots for significant motif–property pairs.

## 3. Quick Start

```bash
# 1. Prepare Conda env with ESM-2 & MEME Suite
conda env create -f environment.yml
conda activate silk-motif

# 2. Run the full pipeline (adjust paths as needed)
python run_pipeline.py \
  --fasta_dir data/windows \
  --validation_csv data/validation/validation_dataset.csv \
  --master_fasta data/fasta/spider-silkome-database.v1.prot.fasta \
  --output_dir results \
  --conda_env silk-motif \
  --generate_plots
