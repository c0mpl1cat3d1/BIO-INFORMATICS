import time
from Bio import Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from io import StringIO
from Bio import SeqIO
import random
import pandas as pd  # <-- NEW IMPORT

Entrez.email = "moulighosh2882003@gmail.com"

# --- Main Configuration ---
MAX_RECORDS_PER_VIRUS = 900
TARGET_SEQUENCES_PER_VIRUS = 300
MAX_N_PERCENTAGE = 3.0
COMBINED_OUTPUT_FILE = "combined_virus_sequences.fasta"
# --- NEW: The "Answer Key" CSV file ---
COMBINED_LABELS_FILE = "combined_virus_labels.csv"

# --- Virus Configuration ---
VIRUS_CONFIG = {
    "Human immunodeficiency virus 1": { "gene_name": "env", "min_length": 2300, "max_length": 2700 },
    "SARS-CoV-2": { "gene_name": "S", "min_length": 3500, "max_length": 4000 },
    "Dengue virus": { "gene_name": "E", "min_length": 1300, "max_length": 1700 },
    "Hepatitis B virus": { "gene_name": "S", "min_length": 600, "max_length": 1300 },
    "Hepatitis C virus": { "gene_name": "E2", "min_length": 900, "max_length": 1200 },
    "Measles morbillivirus": { "gene_name": "H", "min_length": 1700, "max_length": 2100 }
}

# --- Map of full names to the simple labels ---
VIRUS_TO_LABEL_MAP = {
    "Human immunodeficiency virus 1": "hiv",
    "SARS-CoV-2": "sars",
    "Dengue virus": "dengue",
    "Hepatitis B virus": "hepatitis_b",
    "Hepatitis C virus": "hepatitis_c",
    "Measles morbillivirus": "measles"
}

all_final_records = []
# --- NEW: List to hold our label info for the CSV ---
all_label_info = []

print("--- Starting Virus Sequence Download ---")
print(f"Targeting {TARGET_SEQUENCES_PER_VIRUS} sequences per virus.")
print(f"FASTA file: {COMBINED_OUTPUT_FILE}")
print(f"Labels file: {COMBINED_LABELS_FILE}\n")

# --- Loop through each virus ---
for virus_name, config in VIRUS_CONFIG.items():
    gene_name = config['gene_name']
    min_length = config['min_length']
    max_length = config['max_length']
    simple_label = VIRUS_TO_LABEL_MAP[virus_name]

    print(f"Processing: {virus_name} (Label: {simple_label}, Gene: {gene_name})")

    query = (
        f'"{virus_name}"[Organism] AND '
        f'"{gene_name}"[Gene Name] AND '
        f'{min_length}:{max_length}[SLEN]'
    )

    try:
        # 1. Search NCBI
        print("  Searching for record IDs...")
        handle = Entrez.esearch(db="nucleotide", term=query, retmax=MAX_RECORDS_PER_VIRUS)
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]

        if not id_list:
            print("  -> No records found. Skipping.")
            continue

        record_count = len(id_list)
        print(f"  -> Found {record_count} gene sequences. Fetching all...")

        # 2. Fetch all records
        handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="text")
        fasta_data = handle.read()
        handle.close()
        
        # 3. Perform quality control
        high_quality_records = []
        for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
            seq_len = len(record.seq)
            
            if not (min_length <= seq_len <= max_length):
                continue

            n_count = record.seq.upper().count('N')
            n_percentage = (n_count / seq_len) * 100
                                  
            if n_percentage <= MAX_N_PERCENTAGE:
                # --- THIS IS THE CHANGE ---
                # We are NOT modifying the header anymore.
                # We just store the clean ID and label.
                # Note: record.id gives 'MH444177.1'
                all_label_info.append({
                    "sequence_id": record.id,
                    "label": simple_label
                })
                # -------------------------
                high_quality_records.append(record)

        print(f"  -> Quality Control: {len(high_quality_records)}/{record_count} sequences passed.")

        # 4. Sample the high-quality records
        if len(high_quality_records) > TARGET_SEQUENCES_PER_VIRUS:
            print(f"  -> Randomly sampling {TARGET_SEQUENCES_PER_VIRUS} sequences...")
            final_records_to_save = random.sample(high_quality_records, TARGET_SEQUENCES_PER_VIRUS)
        else:
            final_records_to_save = high_quality_records

        # 5. Add to the main list
        all_final_records.extend(final_records_to_save)
        print(f"  ✅ Collected {len(final_records_to_save)} sequences for the combined file.\n")

    except Exception as e:
        print(f"  ❌ An error occurred while processing {virus_name}: {e}\n")

    time.sleep(1) 

print(f"\n--- All processing finished ---")

# --- Write the FASTA file ---
print(f"Total sequences collected: {len(all_final_records)}")
with open(COMBINED_OUTPUT_FILE, "w") as f:
    SeqIO.write(all_final_records, f, "fasta")
print(f"✅ Successfully saved all sequences to '{COMBINED_OUTPUT_FILE}'")

# --- NEW: Write the Labels CSV file ---
# We must filter all_label_info to only include IDs that are in all_final_records
final_ids = {record.id for record in all_final_records}
final_label_data = [info for info in all_label_info if info['sequence_id'] in final_ids]

print(f"Total labels collected: {len(final_label_data)}")
df = pd.DataFrame(final_label_data)
df.to_csv(COMBINED_LABELS_FILE, index=False)
print(f"✅ Successfully saved all labels to '{COMBINED_LABELS_FILE}'")