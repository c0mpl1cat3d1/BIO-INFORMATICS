import pandas as pd
import numpy as np
import os
import joblib  # For saving the label encoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from tensorflow.keras.utils import to_categorical

# --- 1. Load, Combine, and Label Your 3D Data ---
processed_dir = "processed_3d" 

# Update this to match your .npy file names
label_mapping = {
    'dengue_virus_E_protein_embeddings.npy': 'dengue',
    'sars_cov_2_S_protein_embeddings.npy': 'sars_cov_2',
    'hiv_1_env_protein_embeddings.npy': 'hiv',
    'hepatitis_b_S_protein_embeddings.npy': 'hepatitis_b',
    'hepatitis_c_E2_protein_embeddings.npy': 'hepatitis_c',
    'measles_virus_H_protein_embeddings.npy': 'measles'
}

all_X_data = []
all_y_labels = []

print("Loading 3D .npy data...")
for file_name, label in label_mapping.items():
    file_path = os.path.join(processed_dir, file_name)
    try:
        X_data_file = np.load(file_path)
        all_X_data.append(X_data_file)
        
        labels_for_file = [label] * len(X_data_file)
        all_y_labels.extend(labels_for_file)
        
        print(f"  Loaded {file_name} with shape {X_data_file.shape}")
        
    except FileNotFoundError:
        print(f"Warning: Could not find {file_path}. Skipping.")

X = np.vstack(all_X_data)
y = np.array(all_y_labels)

print(f"\nTotal combined 3D data shape (X): {X.shape}")
print(f"Total labels shape (y): {y.shape}")

# --- 2. Encode Labels (e.g., 'dengue' -> 0, 'hiv' -> 1) ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_one_hot = to_categorical(y_encoded, num_classes=len(label_mapping))

# Save the label encoder
joblib.dump(le, 'label_encoder.joblib')
print(f"\nLabel encoder saved to 'label_encoder.joblib'")

# --- 3. Create Train-Test Split (BEFORE oversampling) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, 
    test_size=0.25, 
    random_state=42, 
    stratify=y_one_hot
)

# For SMOTE, we need the 1D encoded labels (not one-hot)
y_train_1d_encoded = np.argmax(y_train, axis=1)

print("\n--- Before Oversampling ---")
original_counts = Counter(y_train_1d_encoded)
print(f"Training set distribution: {original_counts}")
print(f"Label mapping: {list(enumerate(le.classes_))}")

# --- 4. Define and Apply Targeted SMOTE ---
try:
    # !! IMPORTANT: Update this strategy with your class indices and target numbers !!
    # Example: {class_index_1: 2000, class_index_2: 2000, ...}
    # Check the "Label mapping" printout above to find your class indices.
    sampling_strategy = {
        
         2: 2000, # 'hepatitis_c'
        # 3: 2000, # 'hiv'
         4: 2000, # 'measles'
      
    }

    # If strategy is empty, skip SMOTE
    if not sampling_strategy:
        print("\nSampling strategy is empty. Skipping SMOTE.")
        X_train_resampled, y_train_resampled = X_train, y_train
    
    else:
        min_class_size = float('inf')
        for class_idx in sampling_strategy.keys():
            if original_counts[class_idx] > 0:
                min_class_size = min(min_class_size, original_counts[class_idx])

        k = min(5, min_class_size - 1)

        if k < 1:
            print(f"\nError: Smallest class has {min_class_size} sample(s). Cannot apply SMOTE.")
            X_train_resampled, y_train_resampled = X_train, y_train
        else:
            print(f"\nApplying SMOTE with k_neighbors={k}...")
            sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=k)
            
            n_samples, n_timesteps, n_features = X_train.shape
            X_train_2d = X_train.reshape((n_samples, n_timesteps * n_features))
            
            X_train_resampled_2d, y_train_resampled_1d = sm.fit_resample(X_train_2d, y_train_1d_encoded)
            
            new_n_samples = X_train_resampled_2d.shape[0]
            X_train_resampled = X_train_resampled_2d.reshape((new_n_samples, n_timesteps, n_features))
            
            y_train_resampled = to_categorical(y_train_resampled_1d, num_classes=len(label_mapping))
            
            print("\n--- After Targeted Oversampling ---")
            print(f"Resampled training set shape (X): {X_train_resampled.shape}")
            print(f"Resampled training set distribution: {Counter(y_train_resampled_1d)}")

except Exception as e:
    print(f"\nAn error occurred during SMOTE: {e}")
    X_train_resampled, y_train_resampled = X_train, y_train # Default to original data

# --- 5. Save the Processed Data ---
print("\nSaving processed data to .npy files...")

np.save('X_train_resampled.npy', X_train_resampled)
np.save('y_train_resampled.npy', y_train_resampled)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("--- Data Preprocessing Complete ---")
print("Saved files:")
print("- X_train_resampled.npy")
print("- y_train_resampled.npy")
print("- X_test.npy")
print("- y_test.npy")
print("- label_encoder.joblib")