import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os
import pathlib

print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. CONFIGURATION ---
# The single directory containing all class subfolders (with old AND new data)
DATA_DIR = 'dataset' 
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# We only need a few epochs to update the model
EPOCHS = 30 
VALIDATION_SPLIT = 0.2

# --- Path to your PREVIOUSLY trained model ---
OLD_MODEL_PATH = 'waste_classifier_model_cpu.keras' 

# --- Filenames for your NEW, UPDATED model ---
NEW_KERAS_MODEL_PATH = 'waste_classifier_model_cpu_v2.keras'
NEW_TFLITE_MODEL_PATH = 'waste_classifier_quant_int8_cpu_v2.tflite'


# --- 2. Load and Prepare Datasets (Now includes your new images) ---
print("Loading and preparing datasets...")
data_dir_path = pathlib.Path(DATA_DIR)
if not data_dir_path.exists():
    raise FileNotFoundError(f"The directory '{DATA_DIR}' was not found.")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir_path,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir_path,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

class_names = train_dataset.class_names
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. LOAD THE EXISTING MODEL ---
print(f"\nLoading existing model from: {OLD_MODEL_PATH}")
if not os.path.exists(OLD_MODEL_PATH):
    raise FileNotFoundError(f"Could not find the model to update at '{OLD_MODEL_PATH}'")

model = tf.keras.models.load_model(OLD_MODEL_PATH)

# --- 4. RE-COMPILE THE MODEL WITH A LOW LEARNING RATE ---
# This is the most critical step for incremental training.
print("Re-compiling model with a very low learning rate...")
LOW_LEARNING_RATE = 0.00001 # 1e-5
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LOW_LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 5. CONTINUE TRAINING THE MODEL ---
print("\nStarting incremental training on the new data...")

# Use EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)
print("Incremental training complete.")

# --- 6. SAVE THE NEW, UPDATED MODEL ---
print(f"\nSaving updated Keras model to: {NEW_KERAS_MODEL_PATH}")
model.save(NEW_KERAS_MODEL_PATH)
print("Updated model saved.")

# --- 7. CONVERT THE UPDATED MODEL TO TFLITE ---
print(f"\nConverting updated model to TFLite: {NEW_TFLITE_MODEL_PATH}")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
  for input_value, _ in train_dataset.take(100):
    yield [input_value]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

with open(NEW_TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model_quant)
print("Updated TFLite model saved successfully.")