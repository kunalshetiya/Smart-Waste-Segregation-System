import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# --- 1. CONFIGURATION ---
IMAGE_PATH = "WhatsApp Image 2025-09-15 at 20.45.03_a8430f9d.jpg"
MODEL_PATH = "waste_classifier_quant_int8_cpu_v2.tflite"
CLASS_NAMES = ['hazardous', 'organic', 'recyclable']

# --- 2. LOAD THE TFLITE MODEL ---
print(f"Loading model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at '{MODEL_PATH}'")
    sys.exit(1)

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")
    sys.exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print("Model loaded successfully.")

# --- 3. PREPARE THE INPUT IMAGE ---
print(f"Loading and preparing image: {IMAGE_PATH}")
try:
    img = Image.open(IMAGE_PATH).convert('RGB').resize((width, height))
except FileNotFoundError:
    print(f"ERROR: Image file not found at '{IMAGE_PATH}'")
    sys.exit(1)

# --- FIX 1: Input must be uint8 for an INT8 quantized model ---
# The model expects integer pixel values from 0-255, not floats from 0-1.
# We simply create a numpy array from the resized image.
input_tensor = np.expand_dims(img, axis=0)

# --- 4. RUN INFERENCE ---
print("Running inference...")
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()

# Get the raw uint8 output from the model.
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# --- 5. INTERPRET THE OUTPUT & DISPLAY RESULTS ---
# --- FIX 2: De-quantize the uint8 output to get real probabilities ---
# The raw output (0-255) must be converted back to floats using the model's scale and zero-point.
output_scale, output_zero_point = output_details[0]['quantization']
scores_float = (output_data.astype(np.float32) - output_zero_point) * output_scale

# Optional: Apply a softmax function to ensure scores sum to 100%
def softmax(x):
    return np.exp(x) / sum(np.exp(x))
    
scores = softmax(scores_float) # Use 'scores' as the final probabilities

# Find the class with the highest score.
predicted_index = np.argmax(scores)
confidence = scores[predicted_index]
predicted_class_name = CLASS_NAMES[predicted_index]

# Print a detailed report to the console in your desired format.
print("\n--- Prediction Result ---")
print(f"Predicted Class: {predicted_class_name}")
print(f"Confidence:      {confidence:.2%}")
print("\nFull Score Breakdown:")
for i, score in enumerate(scores):
    print(f"  - {CLASS_NAMES[i]:<12}: {score:.2%}")
print("-------------------------\n")