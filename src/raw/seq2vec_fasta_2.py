import os
from Bio import SeqIO
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import argparse


# -----------------------------
# Convert a sequence into overlapping k-mers
# -----------------------------
def get_kmers(seq, k=3):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


# -----------------------------
# Load sequences from a FASTA file
# -----------------------------
def load_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq).upper())
    return sequences


# -----------------------------
# Train Seq2Vec model
# -----------------------------
def train_seq2vec(all_sequences, k=3, vector_size=100, window=5, min_count=1, sg=1):
    sentences = [get_kmers(seq, k) for seq in all_sequences]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model


# -----------------------------
# Convert sequence to embedding
# -----------------------------
def seq2vec(seq, model, k=3):
    kmers = get_kmers(seq, k)
    vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


# -----------------------------
# Main Function
# -----------------------------
def main(input_dir, output_csv, k, vector_size):
    # Collect all fasta files in the folder
    fasta_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.lower().endswith((".fasta", ".fa"))]

    if not fasta_files:
        print("❌ No FASTA files found in the specified directory.")
        return

    print(f"Found {len(fasta_files)} FASTA files.")
    all_sequences = []
    file_labels = []

    # Load all sequences from all FASTA files
    for fasta_file in fasta_files:
        sequences = load_fasta(fasta_file)
        all_sequences.extend(sequences)
        file_labels.extend([os.path.basename(fasta_file)] * len(sequences))
        print(f"Loaded {len(sequences)} sequences from {os.path.basename(fasta_file)}")

    # Train Word2Vec (Seq2Vec) model
    print("Training Seq2Vec model...")
    model = train_seq2vec(all_sequences, k=k, vector_size=vector_size)
    print("✅ Model training completed.")

    # Generate embeddings
    embeddings = [seq2vec(seq, model, k=k) for seq in all_sequences]

    # Save results
    df = pd.DataFrame(embeddings)
    df.insert(0, "source_file", file_labels)
    df.to_csv(output_csv, index=False)

    print(f"✅ Embeddings saved to {output_csv}")


# -----------------------------
# Entry point with arguments
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2Vec embedding generator for multiple FASTA files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing FASTA files.")
    parser.add_argument("--output_csv", default="sequence_embeddings.csv", help="Output CSV file path.")
    parser.add_argument("--k", type=int, default=3, help="k-mer size (default: 3).")
    parser.add_argument("--vector_size", type=int, default=100, help="Embedding dimension (default: 100).")

    args = parser.parse_args()
    main(args.input_dir, args.output_csv, args.k, args.vector_size)
