from Bio import SeqIO
import polars as pl

input_file = "Data/spider-silkome-database.v1.prot.fasta"

# Parse FASTA records
records = list(SeqIO.parse(input_file, "fasta"))

# Extract structured metadata and sequences
parsed = []

for record in records:
    parts = record.description.split("|")
    parsed.append({
        "entry_number": parts[0][1:] if parts[0].startswith(">") else parts[0],
        "tax_id": parts[1] if len(parts) > 1 else None,
        "family": parts[2] if len(parts) > 2 else None,
        "genus": parts[3] if len(parts) > 3 else None,
        "species": parts[4] if len(parts) > 4 else None,
        "gene": parts[5] if len(parts) > 5 else None,
        "protein": parts[6] if len(parts) > 6 else None,
        "region": parts[7] if len(parts) > 7 else None,
        "sequence": str(record.seq)
    })

# Convert to Polars DataFrame
full_proteome_df = pl.DataFrame(parsed)