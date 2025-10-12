from Bio import SeqIO
from gensim.models import Word2Vec
import numpy as np
import pandas as pd


# -----------------------------
# Step 1: Load sequences from FASTA
# -----------------------------
def load_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq).upper())
    return sequences


# -----------------------------
# Step 2: Convert sequences to k-mers
# -----------------------------
def get_kmers(seq, k=3):
    """Convert sequence to list of overlapping k-mers."""
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


# -----------------------------
# Step 3: Train Word2Vec (Seq2Vec) model
# -----------------------------
def train_seq2vec(sequences, k=3, vector_size=100, window=5, min_count=1, sg=1):
    sentences = [get_kmers(seq, k) for seq in sequences]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model


# -----------------------------
# Step 4: Convert each sequence to vector embedding
# -----------------------------
def seq2vec(seq, model, k=3):
    kmers = get_kmers(seq, k)
    vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# -----------------------------
# Step 5: Main Execution
# -----------------------------
if __name__ == "__main__":
    fasta_file = r"C:\Users\SURAJ KUMAR\Desktop\projects\bio-informatics\BIO-INFORMATICS\data\raw\dengue_complete_genomes_2.fasta"
      # <-- replace with your FASTA file name
    sequences = load_fasta(fasta_file)
    print(f"Loaded {len(sequences)} sequences.")

    model = train_seq2vec(sequences, k=3, vector_size=100)
    print("Seq2Vec model trained.")

    embeddings = [seq2vec(seq, model) for seq in sequences]

    # Save embeddings to a CSV file
    df = pd.DataFrame(embeddings)
    df.to_csv("sequence_embeddings.csv", index=False)
    print("Embeddings saved to sequence_embeddings.csv")
