import time
from Bio import Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from io import StringIO
from Bio import SeqIO
import random

Entrez.email = "moulighosh2882003@gmail.com" 


MAX_RECORDS_PER_VIRUS = 9000
TARGET_SEQUENCES_PER_VIRUS = 4000

MAX_N_PERCENTAGE = 3.0 

# Central configuration with a more flexible search query
VIRUS_CONFIG = {
    "SARS-CoV-2": {
        "gene_name": "S",
        "output_file": "sars_cov_2_S_gene.fasta",
        "min_length": 3500  # Spike gene is ~3800 nt
    },
    "Dengue virus": {
        "gene_name": "E",
        "output_file": "dengue_virus_E_gene.fasta",
        "min_length": 1300  # Envelope gene is ~1500 nt
    },
    "Human immunodeficiency virus 1": {
        "gene_name": "env",
        "output_file": "hiv_1_env_gene.fasta",
        "min_length": 2300  # Env gene is ~2500 nt
    },
    "Hepatitis B virus": {
        "gene_name": "P",
        "output_file": "hepatitis_b_P_gene.fasta",
        "min_length": 2300  # Polymerase gene is ~2500 nt
    },
    "Hepatitis C virus": {
        "gene_name": "E2",
        "output_file": "hepatitis_c_E2_gene.fasta",
        "min_length": 900   # E2 gene is ~1000 nt
    },
    "Measles morbillivirus": {
        "gene_name": "H",
        "output_file": "measles_virus_H_gene.fasta",
        "min_length": 1600  # H gene is ~1800 nt
    }}



for virus_name, config in VIRUS_CONFIG.items():
    gene_name = config['gene_name']
    output_file = config['output_file']
    min_length = config['min_length']

    print(f"\n{'='*20}\nProcessing: {virus_name} (Gene: {gene_name})\n{'='*20}")

    # Use a more flexible search query to get more results
    search_query = f'"{virus_name}"[Organism] AND "{gene_name}"[Gene Name]'

    try:
        # 1. Search NCBI to get the list of relevant record IDs
        print("Searching for record IDs with query:")
        print(f"  -> {search_query}")
        
        handle = Entrez.esearch(db="nucleotide",
                                term=search_query,
                                retmax=MAX_RECORDS_PER_VIRUS)
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]

        if not id_list:
            print(f"⚠️ No records found for '{gene_name}' in '{virus_name}'. Skipping.")
            continue

        record_count = len(id_list)
        print(f"Found {record_count} gene sequences. Fetching all in a single request and filtering...")

        # 2. Fetch all records in a single request
        handle = Entrez.efetch(db="nucleotide",
                               id=id_list,
                               rettype="fasta",
                               retmode="text")
        fasta_data = handle.read()
        handle.close()
        
        # 3. Perform quality control on the downloaded data
        high_quality_records = []
        for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
            seq_len = len(record.seq)
            if seq_len < min_length:
                continue 
                 # Skip empty sequences
            
            n_count = record.seq.upper().count('N')
            n_percentage = (n_count / seq_len) * 100
            
            # Keep the sequence only if it's below the 'N' threshold
            if n_percentage <= MAX_N_PERCENTAGE:
                high_quality_records.append(record)

        print(f"\nQuality Control Summary for {virus_name}:")
        print(f"  - Total sequences downloaded: {record_count}")
        print(f"  - Sequences passing QC (>{MAX_N_PERCENTAGE}% Ns removed): {len(high_quality_records)}")

        # 4. Write only the high-quality records to the output file
        if len(high_quality_records) > TARGET_SEQUENCES_PER_VIRUS:
            print(f"  - Randomly sampling {TARGET_SEQUENCES_PER_VIRUS} sequences from the high-quality set.")
            final_records_to_save = random.sample(high_quality_records, TARGET_SEQUENCES_PER_VIRUS)
        else:
            # If we have fewer sequences than the target, just take all of them.
            print(f"  - Taking all {len(high_quality_records)} high-quality sequences as it's less than the target.")
            final_records_to_save= high_quality_records
        with open(output_file, "w") as f:
            SeqIO.write(final_records_to_save, f, "fasta")

        print(f"✅ Successfully saved {len(final_records_to_save)} sequences to '{output_file}'")

    except Exception as e:
        print(f"An error occurred while processing {virus_name}: {e}")

    time.sleep(1)

print(f"\n{'='*20}\nAll downloads complete!\n{'='*20}")