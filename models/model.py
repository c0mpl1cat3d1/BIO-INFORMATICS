import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. Load the Preprocessed Data ---
print("Loading preprocessed data...")
try:
    X_train_resampled = np.load('X_train_resampled.npy')
    y_train_resampled = np.load('y_train_resampled.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
except FileNotFoundError:
    print("Error: Could not find processed .npy files.")
    print("Please run the 'preprocess_data.py' script first.")
    exit()

print(f"Training X shape: {X_train_resampled.shape}")
print(f"Training y shape: {y_train_resampled.shape}")
print(f"Test X shape: {X_test.shape}")
print(f"Test y shape: {y_test.shape}")

# --- 2. Build the CNN+LSTM Model ---
# Get shape info from the loaded data
INPUT_TIMESTEPS = X_train_resampled.shape[1]
INPUT_FEATURES = X_train_resampled.shape[2]
NUM_CLASSES = y_train_resampled.shape[1] # Get num classes from one-hot encoded shape

model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(INPUT_TIMESTEPS, INPUT_FEATURES)))

model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(LSTM(128, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(NUM_CLASSES, activation='softmax'))

model.summary()

# --- 3. Compile and Train ---
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)

# Define callbacks
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    verbose=1, 
    restore_best_weights=True # Saves the best model weights
)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)

model.compile(
    loss='categorical_crossentropy', 
    optimizer=optimizer, 
    metrics=['accuracy']
)

print("\n--- Starting Model Training ---")
history = model.fit(
    X_train_resampled,
    y_train_resampled,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, lr_scheduler]
)
print("--- Training Complete ---")

# --- 4. Save the Best Model ---
# Because restore_best_weights=True, the 'model' object
# now contains the weights from the epoch with the best val_loss.
model_save_path = "best_protein_classifier.keras"
model.save(model_save_path)
print(f"Best model saved to {model_save_path}")

# --- 5. Evaluate the Best Model on Test Data ---
print("\n--- Evaluating model on Test Data ---")
# The model object already has the best weights restored
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")