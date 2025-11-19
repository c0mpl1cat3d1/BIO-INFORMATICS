import time
from Bio import Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from io import StringIO
from Bio import SeqIO
import random



Entrez.email = "moulighosh2882003@gmail.com"





MAX_RECORDS_PER_VIRUS = 9000
TARGET_SEQUENCES_PER_VIRUS = 2000
MAX_N_PERCENTAGE = 3.0





VIRUS_CONFIG = {

    "Human immunodeficiency virus 1": {

        "gene_name": "env",

        "output_file": "hiv_1_env_protein.fasta",

        "min_length": 2300,

        "max_length": 2700

    },

   

    "SARS-CoV-2": {

        "gene_name": "S",

        "output_file": "sars_cov_2_S_protein.fasta",

        "min_length": 3500,

        "max_length": 4000

    },

    "Dengue virus": {

        "gene_name": "E",

        "output_file": "dengue_virus_E_protein.fasta",

        "min_length": 1300,

        "max_length": 1700

    },

   

    "Hepatitis B virus": {

        "gene_name": "S", # (HBsAg)

        "output_file": "hepatitis_b_S_protein.fasta",

        "min_length": 600,  # Adjusted for the S gene's shorter length

        "max_length": 1300

    },

    "Hepatitis C virus": {

        "gene_name": "E2",

        "output_file": "hepatitis_c_E2_protein.fasta",

        "min_length": 900,

        "max_length": 1200

    },

    "Measles morbillivirus": {

        "gene_name": "H",

        "output_file": "measles_virus_H_protein.fasta",

        "min_length": 1700,

        "max_length": 2100

    }

}





for virus_name, config in VIRUS_CONFIG.items():
    gene_name = config['gene_name']
    output_file = config['output_file']
    min_length = config['min_length']



    print(f"\nProcessing: {virus_name} (Gene: {gene_name})\n")



   

    query = (
    f'"{virus_name}"[Organism] AND '
    f'"{config["gene_name"]}"[Gene Name] AND '
    f'{config["min_length"]}:{config["max_length"]}[SLEN]'

)



    try:

        #  Search NCBI to get the list of relevant record IDs

        print("Searching for record IDs with query:")
        print(f"  -> {query}")
        handle = Entrez.esearch(db="nucleotide",
                                term=query,
                                retmax=MAX_RECORDS_PER_VIRUS)

        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]



        if not id_list:
            print(f" No records found for '{gene_name}' in '{virus_name}'. Skipping.")
            continue



        record_count = len(id_list)
        print(f"Found {record_count} gene sequences. Fetching all in a single request and filtering...")



        #  Fetch all records in a single request

        handle = Entrez.efetch(db="nucleotide",
                               id=id_list,
                               rettype="fasta",
                               retmode="text")
        fasta_data = handle.read()
        handle.close()

       

        # Perform quality control 

        high_quality_records = []
        for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
            seq_len = len(record.seq)
            if seq_len < min_length:
                continue

            n_count = record.seq.upper().count('N')
            n_percentage = (n_count / seq_len) * 100
                      

            if n_percentage <= MAX_N_PERCENTAGE:
                high_quality_records.append(record)



        print(f"\nQuality Control Summary for {virus_name}:")
        print(f"  - Total sequences downloaded: {record_count}")
        print(f"  - Sequences passing QC (>{MAX_N_PERCENTAGE}% Ns removed): {len(high_quality_records)}")



        #  Write only the high-quality records 

        if len(high_quality_records) > TARGET_SEQUENCES_PER_VIRUS:
            print(f"  - Randomly sampling {TARGET_SEQUENCES_PER_VIRUS} sequences from the high-quality set.")
            final_records_to_save = random.sample(high_quality_records, TARGET_SEQUENCES_PER_VIRUS)

        else:
            final_records_to_save= high_quality_records
        with open(output_file, "w") as f:
            SeqIO.write(final_records_to_save, f, "fasta")



        print(f"✅ Successfully saved {len(final_records_to_save)} sequences to '{output_file}'")



    except Exception as e:
        print(f"An error occurred while processing {virus_name}: {e}")



    time.sleep(1)



print(f"\nAll downloads complete!\n")