import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report
import os


print("Loading preprocessed data...")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # parent/model
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))  # parent/
DATA_DIR = os.path.join(PARENT_DIR, "data", "processed_80_20_split")
try:
    
    X_train = np.load(f'{DATA_DIR}/train_embeddings.npy') 
    X_test = np.load(f'{DATA_DIR}/test_embeddings.npy')
    
    train_labels_df = pd.read_csv(f'{DATA_DIR}/train_labels.csv')
    test_labels_df = pd.read_csv(f'{DATA_DIR}/test_labels.csv')

    y_train_labels_str = train_labels_df['label']
    y_test_labels_str = test_labels_df['label']

except FileNotFoundError:
    print("Error: Could not find processed .npy or .csv files '.")
    print("Please run the 'preprocess.py' script first.")
    exit()

print(f"Training X shape: {X_train.shape}")
print(f"Training y shape: {y_train_labels_str.shape}")
print(f"Test X shape: {X_test.shape}")
print(f"Test y shape: {y_test_labels_str.shape}")


#  Encode Labels (String -> Integer -> One-Hot) ---
print("\nEncoding labels...")
encoder = LabelEncoder()



# Fit on training data
y_train_int = encoder.fit_transform(y_train_labels_str)
# Transform test data with the same encoder
y_test_int = encoder.transform(y_test_labels_str)

print("Saving label encoder to 'label_encoder.joblib'")
joblib.dump(encoder, "label_encoder.joblib")

print(encoder.classes_)

print("\nSample count per class in TRAIN set:")
print(train_labels_df['label'].value_counts())

print("\nSample count per class in TEST set:")
print(test_labels_df['label'].value_counts())

for virus in ['dengue','hepatitis_b','hepatitis_c','hiv','measles','sars']:
    mask = train_labels_df['label'] == virus
    avg_zero = np.mean(np.mean(X_train[mask] == 0, axis=(1,2)))
    print(virus, avg_zero)



NUM_CLASSES = len(encoder.classes_)
print(f"Found {NUM_CLASSES} classes:")
for i, label in enumerate(encoder.classes_):
    print(f"  {label}: {i}")

# One-hot encode for 'categorical_crossentropy'
y_train_categorical = to_categorical(y_train_int, num_classes=NUM_CLASSES)
y_test_categorical = to_categorical(y_test_int, num_classes=NUM_CLASSES)


#  Calculate Class Weights ---
print("\nCalculating class weights...")
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train_int),  
    y=y_train_int                   
)

# Convert to a dictionary that Keras understands (keys are integers 0, 1, 2...)
class_weights_dict = dict(zip(np.unique(y_train_int), weights))
print("Class weights dictionary:")
print(class_weights_dict)



INPUT_TIMESTEPS = X_train.shape[1]   # max sequence length 
INPUT_FEATURES = X_train.shape[2]    # vector dimension

model = Sequential()
# Masking layer to ignore 0.0 padding
model.add(Masking(mask_value=0.0, input_shape=(INPUT_TIMESTEPS, INPUT_FEATURES)))

# Convolutional block
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Recurrent block
model.add(LSTM(128, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Dense classification block
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.summary()



optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    verbose=1, 
    restore_best_weights=True
)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)
model_checkpoint = ModelCheckpoint(
    filepath="sequence_classifier2.keras", 
    monitor='val_loss',
    save_best_only=True, 
    verbose=1
)

model.compile(
    loss='categorical_crossentropy', 
    optimizer=optimizer, 
    metrics=['accuracy']
)

#  Train the Model (with Class Weights) ---
print("\n--- Starting Model Training ---")
history = model.fit(
    X_train,
    y_train_categorical,  # Use the one-hot encoded labels
    validation_data=(X_test, y_test_categorical),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, lr_scheduler, model_checkpoint],
    class_weight=class_weights_dict 
)
print("--- Training Complete ---")



print("\n--- Evaluating Best Model on Test Data ---")

# Get overall loss and accuracy (the same as before)
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=1)
print(f"\nOverall Test Loss: {test_loss:.4f}")
print(f"Overall Test Accuracy: {test_accuracy:.4f}")


print("\n--- Detailed Classification Report ---")


y_pred_probs = model.predict(X_test)


y_pred_int = np.argmax(y_pred_probs, axis=1)

# We already have the true integer labels in y_test_int
# We get the class names (e.g., "dengue", "hiv") from the encoder
class_names = encoder.classes_

print(classification_report(y_test_int, y_pred_int, target_names=class_names, zero_division=0))


#  Plot Training History
print("\n--- Plotting Training History ---")
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

print("Script finished.")