import time
from Bio import Entrez

# --- Parameters ---
Entrez.email = "moulighosh2882003@gmail.com"
MAX_RECORDS_PER_VIRUS = 9000

# Central configuration dictionary with virus-specific length ranges
VIRUS_CONFIG = {
     
    "Dengue virus": {
        "filename": "dengue_virus_genomes.fasta",
        "length_range": "10000:12000"
    },

  
    "Hepatitis C virus": {
        "filename": "hepatitis_c_genomes.fasta",
        "length_range": "8500:11500"
    },
   
    "Measles morbillivirus": {
        "filename": "measles_virus_genomes.fasta",
        "length_range": "15000:16000"  # ~15.9kb genome
    }

}

# --- Main Script ---

# Loop through each virus in the configuration
for virus_name, config in VIRUS_CONFIG.items():
    output_file = config['filename']
    length_filter = config['length_range']

    

    # Dynamically build the search query
    search_query = f'"{virus_name}"[Organism] AND {length_filter}[SLEN]'

    try:
        # 1. Search for record IDs
        print("Searching for record IDs with query:")
        print(f"  -> {search_query}")
        
        handle = Entrez.esearch(db="nucleotide",
                                term=search_query,
                                retmax=MAX_RECORDS_PER_VIRUS)
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]

        if not id_list:
            print(f"⚠️ No records found for '{virus_name}'. Skipping.")
            continue

        record_count = len(id_list)
        print(f"Found {record_count} IDs. Fetching all records in a single request...")

        # 2. Fetch all records at once (NO BATCHING)
        handle = Entrez.efetch(db="nucleotide",
                               id=id_list,
                               rettype="fasta",
                               retmode="text")
        fasta_data = handle.read()
        handle.close()

        # 3. Write data to file
        print(f"Writing {record_count} sequences to '{output_file}'...")
        with open(output_file, "w") as f:
            f.write(fasta_data)

        print(f"✅ Successfully downloaded sequences to '{output_file}'")

    except Exception as e:
        print(f"❌ An error occurred while processing {virus_name}: {e}")

    # Be polite to NCBI servers
    time.sleep(1)

print(f"\n{'='*20}\nAll downloads complete!\n{'='*20}")