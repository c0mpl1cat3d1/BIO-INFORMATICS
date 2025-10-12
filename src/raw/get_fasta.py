
from Bio import Entrez

Entrez.email = "amazingsuraj101@gmail.com"

# 3. Set your search query and output file name
# You can change this to download data for other viruses.
search_query = '"Dengue virus"[Organism] AND "complete genome"[Title]'
output_file = "dengue_complete_genomes.fasta"
num_records = 10000 # Start with a small number for testing

# --- End of Parameters ---


# 4. Search the NCBI Nucleotide database
print(f"Searching NCBI for {num_records} records matching: '{search_query}'...")
handle = Entrez.esearch(db="nucleotide",
                        term=search_query,
                        retmax=num_records)
record = Entrez.read(handle)
handle.close()

id_list = record["IdList"]
if not id_list:
    print("No records found for the search query.")
    exit()

print(f"Found {len(id_list)} record IDs. Fetching the FASTA files...")

# 5. Fetch the records in FASTA format
handle = Entrez.efetch(db="nucleotide",
                        id=id_list,
                        rettype="fasta",
                        retmode="text")
fasta_data = handle.read()
handle.close()

# 6. Save the data to a file
with open(output_file, "w") as f:
    f.write(fasta_data)

print(f"✅ Successfully downloaded sequences to '{output_file}'")