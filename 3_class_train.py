import tensorflow as tf
import keras
import os
import pathlib

print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Force CPU-Only Execution ---
# CPU-ONLY: Hide all GPUs from TensorFlow.
# This ensures the script will run on the CPU even if a GPU is installed.
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
    print("Successfully configured for CPU-only execution.")
except:
    print("Could not hide GPUs. May still run on GPU if available.")


# --- 2. Configuration Parameters ---
DATA_DIR = 'dataset'
IMG_SIZE = (160, 160)

# CPU-ONLY: Smaller batch sizes are generally better for CPU training.
BATCH_SIZE = 32

EPOCHS = 50
VALIDATION_SPLIT = 0.2

# --- 3. Load and Prepare Datasets ---
print("Loading and preparing datasets...")
data_dir_path = pathlib.Path(DATA_DIR)
if not data_dir_path.exists():
    raise FileNotFoundError(f"The directory '{DATA_DIR}' was not found. Please check the path.")

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
print(f"Found classes: {class_names}")
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. Build the Model ---
print("Building the model...")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
], name="data_augmentation")

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
# CPU-ONLY: Removed dtype='float32' as it was for mixed precision stability
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# --- 5. Compile the Model ---
print("Compiling the model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    # CPU-ONLY: Removed jit_compile=True as it's a GPU/TPU optimization
)

model.summary()

# --- 6. Train the Model ---
print(f"\nStarting model training on CPU with BATCH_SIZE={BATCH_SIZE}...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)
print("Model training complete.")

# --- 7. Display Training Summary ---
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("\n--- Training Summary Report ---")
print(f"Training completed after {EPOCHS} epochs.")
print(f"  - Final Training Accuracy:   {final_train_acc:.2%}")
print(f"  - Final Validation Accuracy: {final_val_acc:.2%}")
print(f"  - Final Training Loss:       {final_train_loss:.4f}")
print(f"  - Final Validation Loss:     {final_val_loss:.4f}")
print("---------------------------------\n")

# --- 8. Save the Keras Model ---
keras_model_filename = 'waste_classifier_model_cpu.keras'
print(f"Saving the trained Keras model as {keras_model_filename}...")
model.save(keras_model_filename)
print("Model saved.")

# --- 9. Convert to TFLite with Full Integer Quantization (INT8) ---
print("\nConverting to TensorFlow Lite with 8-bit integer quantization...")
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

tflite_model_filename = 'waste_classifier_quant_int8_cpu.tflite'
with open(tflite_model_filename, 'wb') as f:
    f.write(tflite_model_quant)

print(f"Successfully converted and saved quantized model as {tflite_model_filename}")
print(f"File size: {os.path.getsize(tflite_model_filename) / 1024:.2f} KB")