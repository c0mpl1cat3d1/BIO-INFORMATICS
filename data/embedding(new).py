import os
from Bio import SeqIO
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Get script directory
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.path.abspath(".")
    print(f"Warning: __file__ not defined. Using current directory: {script_dir}")


# -----------------------------
# Helper Functions
# -----------------------------

def load_fasta(file_path):
    """Loads all sequences from a single FASTA file. Returns list of (id, sequence) tuples."""
    sequences = []
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append((record.id, str(record.seq).upper()))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return sequences


def get_kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def train_seq2vec(sequences_with_ids, k=3, vector_size=100, window=5, min_count=1, sg=1):
    seq_list = [seq for seq_id, seq in sequences_with_ids]
    sentences = [get_kmers(seq, k) for seq in seq_list]
    print(f"Training Word2Vec on {len(sentences)} sequences...")
    return Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)


def seq_to_vectors_list(seq, model, k=3):
    kmers = get_kmers(seq, k)
    vectors = [model.wv[k] for k in kmers if k in model.wv]
    if len(vectors) == 0:
        return [np.zeros(model.vector_size)]
    return vectors 


# -----------------------------
# SLIDING WINDOW FIX
# -----------------------------

def windows_from_vectors(vector_list, window_size=512, stride=256):
    """Break sequence vectors into fixed-size windows."""
    windows = []
    n = len(vector_list)

    if n < window_size:
        # pad ONE window
        padded = vector_list + [np.zeros_like(vector_list[0])] * (window_size - n)
        return [padded]

    for start in range(0, n - window_size + 1, stride):
        win = vector_list[start:start + window_size]
        windows.append(win)

    # Ensure last window included
    if (n - window_size) % stride != 0:
        windows.append(vector_list[-window_size:])

    return windows


# -----------------------------
# Main Script
# -----------------------------

if __name__ == "__main__":

    base_raw_path = os.path.join(script_dir, "raw")

    FILE_LABEL_MAP = {
        "dengue_virus_E_protein.fasta": "dengue",
        "sars_cov_2_S_protein.fasta": "sars",
        "hiv_1_env_protein.fasta": "hiv",
        "hepatitis_b_S_protein.fasta": "hepatitis_b",
        "hepatitis_c_E2_protein.fasta": "hepatitis_c",
        "measles_virus_H_protein.fasta": "measles"
    }

    OUTPUT_DIR = os.path.join(script_dir, "processed_80_20_split")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    KMER_SIZE = 3
    VECTOR_DIM = 100
    TEST_SPLIT_SIZE = 0.20

    # -----------------------------
    # PHASE 1 — Load Sequences
    # -----------------------------

    print("\n--- Phase 1: Loading All Sequences ---")

    all_sequences = []
    all_labels = []

    for filename, label in FILE_LABEL_MAP.items():
        fpath = os.path.join(base_raw_path, filename)
        print(f"Loading {label} from {filename}...")
        seqs = load_fasta(fpath)

        for sid, seq in seqs:
            all_sequences.append((sid, seq, label))
            all_labels.append(label)

    print(f"Total sequences loaded: {len(all_sequences)}")


    # -----------------------------
    # PHASE 2 — Stratified Split
    # -----------------------------

    train_data, test_data = train_test_split(
        all_sequences, test_size=TEST_SPLIT_SIZE, stratify=all_labels, random_state=42
    )

    print(f"Training sequences: {len(train_data)}")
    print(f"Testing sequences:  {len(test_data)}")


    # -----------------------------
    # PHASE 3 — Train Word2Vec
    # -----------------------------

    print("\n--- Phase 3: Training Word2Vec ---")
    train_for_w2v = [(sid, seq) for sid, seq, lbl in train_data]
    model = train_seq2vec(train_for_w2v, k=KMER_SIZE, vector_size=VECTOR_DIM)
    model.save("word2vec.model")
    print("Saved Word2Vec model.")


    # -----------------------------
    # PHASE 4 — Embed + Sliding Windows
    # -----------------------------

    print("\n--- Phase 4: Generating Window-Based Embeddings ---")

    def process_and_save_set(dataset, name):
        print(f"\nProcessing {name} set...")

        window_vectors = []
        new_rows = []

        for sid, seq, label in dataset:
            vec_list = seq_to_vectors_list(seq, model, k=KMER_SIZE)

            windows = windows_from_vectors(vec_list, window_size=512, stride=256)

            for w in windows:
                window_vectors.append(w)
                new_rows.append({
                    "sequence_id": sid,
                    "label": label,
                    "original_sequence_length": len(seq)
                })

        # Convert to numpy array: (#windows, 512, 100)
        window_vectors = np.array(window_vectors, dtype=np.float32)

        # Save embeddings
        np.save(os.path.join(OUTPUT_DIR, f"{name}_embeddings.npy"), window_vectors)

        # Save labels
        df = pd.DataFrame(new_rows)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_labels.csv"), index=False)

        print(f"Saved {name}_embeddings.npy → shape {window_vectors.shape}")
        print(f"Saved {name}_labels.csv → {len(df)} rows")

    process_and_save_set(train_data, "train")
    process_and_save_set(test_data, "test")

    print("\nAll files generated successfully with sliding windows!")
