import os
from Bio import SeqIO
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
# --- NEW IMPORT ---
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Get the absolute path to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

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
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model


# -----------------------------
# STEP 4: MODIFIED FUNCTION
# -----------------------------
def seq_to_vectors_list(seq, model, k=3):
    """
    Converts a single sequence string into a LIST of k-mer vectors.
    It does NOT average them.
    """
    kmers = get_kmers(seq, k)
    vectors = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
    if len(vectors) == 0:
        # Return a list containing one zero vector if no k-mers are found
        return [np.zeros(model.vector_size)] 
    return vectors


# ===================================================================
# MAIN EXECUTION (HEAVILY MODIFIED)
# ===================================================================
if __name__ == "__main__":
    
    # 1. Define your list of 6 input FASTA files
    fasta_files_list = [
        os.path.join(script_dir, r"raw\dengue_virus_E_protein.fasta"),
        os.path.join(script_dir, r"raw\sars_cov_2_S_protein.fasta"),
        os.path.join(script_dir, r"raw\hiv_1_env_protein.fasta"),
        os.path.join(script_dir, r"raw\hepatitis_b_S_protein.fasta"),
        os.path.join(script_dir, r"raw\hepatitis_c_E2_protein.fasta"), # Make sure this filename is correct
        os.path.join(script_dir, r"raw\measles_virus_H_protein.fasta") # Make sure this filename is correct
    ]
    
    # 2. Define a NEW output directory
    OUTPUT_DIR = os.path.join(script_dir, "processed_3d")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. Define your model parameters
    KMER_SIZE = 3
    VECTOR_DIM = 100
    
    # --- NEW: Define a MAX_LENGTH for padding ---
    # This pads all sequences to have 2000 k-mers.
    # Adjust this based on your data (e.g., length of your longest sequence)
    MAX_SEQUENCE_LENGTH = 2000 

    # --- Phase 1: Train ONE Combined Model ---
    all_sequences_with_ids = []
    for file_path in fasta_files_list:
        seqs_from_file = load_fasta(file_path)
        all_sequences_with_ids.extend(seqs_from_file)
    
    print(f"Total sequences for training: {len(all_sequences_with_ids)}")

    if not all_sequences_with_ids:
        print("No sequences loaded. Check file paths. Exiting.")
    else:
        model = train_seq2vec(all_sequences_with_ids, k=KMER_SIZE, vector_size=VECTOR_DIM)
        print("Combined Seq2Vec model trained.")

        # --- Phase 2: Generate and Save 3D Padded Data ---
        print("\n--- Phase 2: Generating 3D Padded Embedding Files ---")
        
        for file_path in fasta_files_list:
            print(f"Processing: {file_path}")
            current_file_seqs = load_fasta(file_path)
            
            if not current_file_seqs: continue

            all_vector_lists = []
            all_sequence_ids = []
            
            for seq_id, seq in current_file_seqs:
                # Get the list of vectors (e.g., shape (498, 100))
                vector_list = seq_to_vectors_list(seq, model, k=KMER_SIZE)
                all_vector_lists.append(vector_list)
                all_sequence_ids.append(seq_id)

            # --- This is the key padding step ---
            padded_data = pad_sequences(
                all_vector_lists,
                maxlen=MAX_SEQUENCE_LENGTH,
                dtype='float32',
                padding='post',    # Add 0s to the end
                truncating='post'  # Cut from the end
            )
            
            # 'padded_data' is now a 3D NumPy array (e.g., (50, 2000, 100))
            print(f"  -> Created 3D array with shape: {padded_data.shape}")

            # --- Save as .npy file (not CSV) ---
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{base_name}_embeddings.npy"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            np.save(output_path, padded_data)
            
            # --- Save the IDs in a separate .txt file ---
            id_filename = f"{base_name}_ids.txt"
            id_output_path = os.path.join(OUTPUT_DIR, id_filename)
            with open(id_output_path, 'w') as f:
                for seq_id in all_sequence_ids:
                    f.write(f"{seq_id}\n")
                    
            print(f"  -> Saved 3D data to {output_path}")

        print("All files processed.")