import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from gensim.models import Word2Vec
from Bio import SeqIO
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings

# --- Suppress Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='gensim')

# --- 1. DEFINE ALL FILE PATHS ---

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath(".")

# All files are inside data/
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

W2V_MODEL_PATH = os.path.join(DATA_DIR, "word2vec.model")
ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.joblib")
KERAS_MODEL_PATH = os.path.join(DATA_DIR, "sequence_classifier2.keras")
FASTA_FILE_PATH = os.path.join(DATA_DIR, "combined_virus_sequences.fasta")
LABELS_CSV_PATH = os.path.join(DATA_DIR, "combined_virus_labels.csv")

# --- CONFIGURATION ---
KMER_SIZE = 3
VECTOR_DIM = 100
MAX_SEQUENCE_LENGTH = 512
WINDOW_STRIDE = 256

# --- HELPER FUNCTIONS ---

def get_kmers(seq, k=KMER_SIZE):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def sliding_windows(seq, window_size=MAX_SEQUENCE_LENGTH, stride=WINDOW_STRIDE):
    for i in range(0, len(seq) - window_size + 1, stride):
        yield seq[i:i + window_size]
    if len(seq) < window_size:
        yield seq  # short sequences

def seq_to_window_vectors(seq, w2v_model):
    window_vectors = []
    for window in sliding_windows(seq):
        kmers = get_kmers(window)
        vectors = [w2v_model.wv[k] for k in kmers if k in w2v_model.wv]
        if len(vectors) == 0:
            vectors = [np.zeros(w2v_model.vector_size)]
        window_vectors.append(vectors)
    return window_vectors

# --- MAIN FUNCTION ---

def predict_on_combined_fasta():
    print(f"--- Starting Prediction ---")
    try:
        print("Loading models and label file...")
        w2v_model = Word2Vec.load(W2V_MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        labels_df = pd.read_csv(LABELS_CSV_PATH)
        label_lookup = pd.Series(labels_df.label.values, index=labels_df.sequence_id).to_dict()
    except FileNotFoundError as e:
        print(f"\n--- ERROR: FILE NOT FOUND ---\nMissing file: {e.filename}")
        return
    except Exception as e:
        print(f"Error loading models or data: {e}")
        return

    print(f"\nModel classes: {encoder.classes_}")
    print(f"\nProcessing sequences from {FASTA_FILE_PATH}...")

    y_true_labels, y_pred_labels = [], []
    total_sequences, unlabeled_sequences = 0, 0

    for record in SeqIO.parse(FASTA_FILE_PATH, "fasta"):
        total_sequences += 1
        true_label = label_lookup.get(record.id)
        if true_label is None:
            unlabeled_sequences += 1
            continue

        seq = str(record.seq).upper()
        windows = seq_to_window_vectors(seq, w2v_model)
        window_preds = []

        for w in windows:
            X_pred = pad_sequences([w], maxlen=MAX_SEQUENCE_LENGTH, dtype='float32', padding='post', truncating='post')
            prob = model.predict(X_pred, verbose=0)[0]
            window_preds.append(prob)

        avg_pred = np.mean(window_preds, axis=0)
        pred_int = np.argmax(avg_pred)
        pred_label = encoder.inverse_transform([pred_int])[0]

        y_true_labels.append(true_label)
        y_pred_labels.append(pred_label)

    print(f"\nProcessed: {total_sequences}, Skipped: {unlabeled_sequences}, Evaluated: {len(y_true_labels)}")
    if len(y_true_labels) == 0:
        print("No sequences matched. Exiting.")
        return

    print("\n--- Overall Results ---")
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, labels=encoder.classes_, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=encoder.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved to 'confusion_matrix.png'")

if __name__ == "__main__":
    predict_on_combined_fasta()
