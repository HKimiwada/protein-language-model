from exploratory_data_analysis import proteome_eda_df
import polars as pl

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