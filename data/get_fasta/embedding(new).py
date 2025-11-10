import os
from Bio import SeqIO
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re

# Get the absolute path to the directory where this script is located
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.path.abspath(".")
    print(f"Warning: __file__ not defined. Using current directory: {script_dir}")

# --- Helper Functions (Unchanged) ---
def load_fasta(file_path):
    """Loads all sequences from a single FASTA file. Returns list of (id, sequence) tuples."""
    sequences = []
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append((record.id, str(record.seq).upper()))
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Skipping.")
    except Exception as e:
        print(f"Warning: Error processing {file_path}: {e}. Skipping.")
    return sequences

def get_kmers(seq, k=3):
    """Convert sequence to list of overlapping k-mers."""
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def train_seq2vec(sequences_with_ids, k=3, vector_size=100, window=5, min_count=1, sg=1):
    """Trains the Word2Vec model on a list of (id, seq) tuples."""
    seq_list = [seq for seq_id, seq in sequences_with_ids]
    sentences = [get_kmers(seq, k) for seq in seq_list]
    print(f"Training Word2Vec on {len(sentences)} sequences...")
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model

def seq_to_vectors_list(seq, model, k=3):
    """Converts a single sequence string into a LIST of k-mer vectors."""
    kmers = get_kmers(seq, k)
    vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
    if len(vectors) == 0:
        return [np.zeros(model.vector_size)] 
    return vectors

# --- DELETED get_label_from_filename function ---

if __name__ == "__main__":
    
    base_raw_path = os.path.join(script_dir, "raw")
    
    # --- NEW: Explicit mapping of filenames to labels ---
    FILE_LABEL_MAP = {
        "dengue_virus_E_protein.fasta": "dengue",
        "sars_cov_2_S_protein.fasta": "sars",
        "hiv_1_env_protein.fasta": "hiv",
        "hepatitis_b_S_protein.fasta": "hepatitis_b", # <-- THE FIX
        "hepatitis_c_E2_protein.fasta": "hepatitis_c", # <-- THE FIX
        "measles_virus_H_protein.fasta": "measles"
    }
    
    OUTPUT_DIR = os.path.join(script_dir, "processed_80_20_split")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    KMER_SIZE = 3
    VECTOR_DIM = 100
    MAX_SEQUENCE_LENGTH = 2000 
    TEST_SPLIT_SIZE = 0.20 # 80-20 split

    # --- STEP 1: Load *ALL* data first (MODIFIED) ---
    print("--- Phase 1: Loading All Sequences ---")
    
    all_sequences = [] # List of tuples: (seq_id, sequence, label)
    all_labels = []      # List of just labels, for stratification
    
    # --- MODIFIED LOOP to use the map ---
    for filename, label in FILE_LABEL_MAP.items():
        file_path = os.path.join(base_raw_path, filename)
        print(f"Loading {label} from {filename}...")
        
        seqs_from_file = load_fasta(file_path)
        if not seqs_from_file:
            print(f"Warning: No sequences loaded from {filename}. Skipping.")
            continue
            
        for seq_id, seq in seqs_from_file:
            all_sequences.append((seq_id, seq, label))
            all_labels.append(label)
            
    print(f"\nTotal sequences loaded: {len(all_sequences)}")
    
    if not all_sequences:
        print("No sequences loaded. Check file paths. Exiting.")
        exit()

    # --- STEP 2: Create the 80/20 Train/Test Split (Unchanged) ---
    print(f"\n--- Phase 2: Splitting data {1-TEST_SPLIT_SIZE:.0%}/{TEST_SPLIT_SIZE:.0%} (Stratified) ---")
    train_data, test_data = train_test_split(
        all_sequences,
        test_size=TEST_SPLIT_SIZE,
        stratify=all_labels,
        random_state=42
    )
    print(f"Training sequences: {len(train_data)}")
    print(f"Testing sequences:  {len(test_data)}")
    
    # --- STEP 3: Train Word2Vec (Unchanged) ---
    print("\n--- Phase 3: Training Word2Vec Model ---")
    train_seqs_for_w2v = [(seq_id, seq) for seq_id, seq, label in train_data]
    model = train_seq2vec(train_seqs_for_w2v, k=KMER_SIZE, vector_size=VECTOR_DIM)
    print("Word2Vec model trained *only* on training data.")
    
    # --- ADDING THE SAVE LINE (from our previous discussion) ---
    model.save("word2vec.model")
    print("Word2Vec model successfully saved to 'word2vec.model'")
    
    # --- STEP 4: Process & Save Sets (Unchanged) ---
    print("\n--- Phase 4: Generating and Saving Embeddings ---")
    
    def process_and_save_set(dataset, set_name):
        print(f"Processing {set_name} set...")
        all_vector_lists = []
        labels_and_ids = [] 
        
        for seq_id, seq, label in dataset:
            vector_list = seq_to_vectors_list(seq, model, k=KMER_SIZE)
            all_vector_lists.append(vector_list)
            labels_and_ids.append({
                "sequence_id": seq_id,
                "label": label,
                "original_sequence_length": len(seq)
            })
            
        padded_data = pad_sequences(
            all_vector_lists,
            maxlen=MAX_SEQUENCE_LENGTH,
            dtype='float32',
            padding='post',    
            truncating='post' 
        )
        
        npy_filename = f"{set_name}_embeddings.npy"
        npy_path = os.path.join(OUTPUT_DIR, npy_filename)
        np.save(npy_path, padded_data)
        
        csv_filename = f"{set_name}_labels.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)
        df = pd.DataFrame(labels_and_ids)
        df.to_csv(csv_path, index=False)
        
        print(f"  -> Saved {set_name} embeddings to {npy_path} (Shape: {padded_data.shape})")
        print(f"  -> Saved {set_name} labels to {csv_path} (Rows: {len(df)})")

    process_and_save_set(train_data, "train")
    process_and_save_set(test_data, "test")
    
    print("\nAll files processed correctly (no data leakage).")