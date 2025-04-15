from data_loading import full_proteome_df
import polars as pl
import matplotlib.pyplot as plt
proteome_eda_df = full_proteome_df.select(pl.col("family", "genus", "species", "protein", "sequence"))

# -------------------------------
# 1. Data Quality and Integrity
# -------------------------------
proteome_eda_df.null_count()

# Alternatively, check for empty strings in critical columns (if applicable)
for col in ["family", "genus", "species", "protein", "sequence"]:
    missing_empty = proteome_eda_df.filter(pl.col(col) == "").height
    print(f"Empty string count in column '{col}':", missing_empty)

# Identify duplicate rows.
# Calculate the number of duplicate rows based on all columns.
unique_df = proteome_eda_df.unique()
num_duplicates = proteome_eda_df.height - unique_df.height
print("\nTotal rows:", proteome_eda_df.height)
print("Unique rows:", unique_df.height)
print("Duplicate row count:", num_duplicates)

# Review the unique values in key taxonomy columns.
print("\nUnique families:", proteome_eda_df["family"].unique())
print("Unique genera:", proteome_eda_df["genus"].unique())
print("Unique species:", proteome_eda_df["species"].unique())

# -------------------------------
# 2. Descriptive Statistics & Basic Distributions
# -------------------------------

# Create a new column for protein sequence length
# Using built-in string length method; this converts the sequence column to its length (an integer)
proteome_eda_df = proteome_eda_df.with_columns(
    pl.col("sequence").str.len_chars().alias("sequence_length")
)
print("\nDataFrame with 'sequence_length' column added:")
print(proteome_eda_df.head())

# Summarize the sequence length statistics
sequence_length_stats = proteome_eda_df.select(pl.col("sequence_length")).describe()
print("\nSequence Length Summary Statistics:")
print(sequence_length_stats)

# Distribution of proteins by taxonomic family
family_counts = proteome_eda_df.group_by("family").agg(pl.len().alias("count")).sort("count", descending=True)
print("\nProtein counts by family:")
print(family_counts)

# Unique protein types
protein_counts = proteome_eda_df.group_by("protein").agg(pl.len().alias("count")).sort("count", descending=True)
print("\nUnique protein types:") # Show everything
print(protein_counts)

# -------------------------------
# Visualizations using matplotlib
# -------------------------------

# Plot a histogram to visualize the distribution of sequence lengths
plt.figure(figsize=(8, 5))
plt.hist(proteome_eda_df["sequence_length"].to_list(), bins=20, edgecolor='black')
plt.xlabel("Protein Sequence Length")
plt.ylabel("Frequency")
plt.title("Distribution of Protein Sequence Lengths")
plt.tight_layout()
plt.show()

# Bar chart for the distribution of proteins by family.
# Convert polars LazyFrame/Series to lists for plotting.
plt.figure(figsize=(10, 6))
plt.bar(family_counts["family"].to_list(), family_counts["count"].to_list())
plt.xlabel("Family")
plt.ylabel("Protein Count")
plt.title("Protein Count by Family")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
