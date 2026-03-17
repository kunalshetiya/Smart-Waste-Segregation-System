import RPi.GPIO as GPIO
import onnxruntime as ort
import numpy as np
import cv2
import time

# ###############################################################################
# ## CONFIGURATION - UPDATED 17-SEPT-2025                                      ##
# ###############################################################################

# --- GPIO Pin setup (keep as per your wiring) ---
MOTOR_IN1 = 24
MOTOR_IN2 = 23
MOTOR_IN3 = 25
MOTOR_IN4 = 8
SERVO_PIN = 18

# --- Model and Camera setup ---
VIDEO_URL = "http://10.24.102.14:8080/video"
MODEL_PATH = "model.onnx"
CONFIDENCE_THRESHOLD = 0.60

ORIGINAL_CLASS_NAMES = [
    'Battery', 'Biological', 'Cardboard', 'Clothes', 'Glass',
    'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash'
]

# <<< CHANGED: Updated servo position for 'metal'
SERVO_POSITIONS = {
    'plastic': 30,
    'paper': 60,
    'metal': 180, # Set to 180 degrees for pass-through
    'glass': 120,
    'unclassified': 150,
}

# <<< CHANGED: Set a fixed neutral position, as 'metal' is now an extreme angle
SERVO_NEUTRAL_POSITION = 90

# ###############################################################################

# --- GPIO and helper functions (unchanged) ---
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
# ... (GPIO setup is the same) ...
GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)
GPIO.setup(MOTOR_IN3, GPIO.OUT)
GPIO.setup(MOTOR_IN4, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

def run_conveyor():
    print("--> Starting conveyor...")
    GPIO.output(MOTOR_IN1, GPIO.HIGH); GPIO.output(MOTOR_IN2, GPIO.LOW)
    GPIO.output(MOTOR_IN3, GPIO.HIGH); GPIO.output(MOTOR_IN4, GPIO.LOW)

def stop_conveyor():
    print("--> Stopping conveyor...")
    GPIO.output(MOTOR_IN1, GPIO.LOW); GPIO.output(MOTOR_IN2, GPIO.LOW)
    GPIO.output(MOTOR_IN3, GPIO.LOW); GPIO.output(MOTOR_IN4, GPIO.LOW)

def set_servo_angle(angle):
    print(f"--> Moving servo to {angle} degrees.")
    duty = angle / 18 + 2
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)

def classify_image(session, image):
    # This function is unchanged
    input_name = session.get_inputs()[0].name
    _, _, height, width = session.get_inputs()[0].shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image_rgb, (width, height))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)
    input_tensor = np.expand_dims(img_array, axis=0)
    output_name = session.get_outputs()[0].name
    results = session.run([output_name], {input_name: input_tensor})
    output_data = results[0][0]
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    original_scores = softmax(output_data)
    grouped_scores = {'metal': 0.0, 'glass': 0.0, 'paper': 0.0, 'plastic': 0.0, 'unclassified': 0.0}
    for i, class_name in enumerate(ORIGINAL_CLASS_NAMES):
        score = original_scores[i]
        name_lower = class_name.lower()
        if name_lower == 'metal': grouped_scores['metal'] += score
        elif name_lower == 'glass': grouped_scores['glass'] += score
        elif name_lower in ['paper', 'cardboard']: grouped_scores['paper'] += score
        elif name_lower == 'plastic': grouped_scores['plastic'] += score
        else: grouped_scores['unclassified'] += score
    predicted_class_name = max(grouped_scores, key=grouped_scores.get)
    boost_factor = 1.25
    original_confidence = grouped_scores[predicted_class_name]
    boosted_confidence = min(original_confidence * boost_factor, 1.0)
    grouped_scores[predicted_class_name] = boosted_confidence
    final_confidence = grouped_scores[predicted_class_name]
    return predicted_class_name, final_confidence

# --- Main Program ---
try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("Model loaded successfully.")
    set_servo_angle(SERVO_NEUTRAL_POSITION)
    print(f"System Initialized. Servo is in the neutral position ({SERVO_NEUTRAL_POSITION} degrees).")

    while True:
        input("\n[STEP 1] Place an object on the belt, then press Enter to begin...")
        
        # ... (Camera capture and classification logic is unchanged) ...
        camera = cv2.VideoCapture(VIDEO_URL)
        ret, frame = camera.read()
        camera.release()
        if not ret: continue
        category, confidence = classify_image(session, frame)
        print("\n---------------------------------\n      Prediction Result\n---------------------------------")
        print(f"Material:   {category}\nConfidence: {confidence:.2%}")
        print("---------------------------------")
        
        final_category = category
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Confidence below threshold. Defaulting to 'unclassified' path.")
            final_category = 'unclassified'
            
        print("\n[STEP 3] Running sorting sequence...")
        target_angle = SERVO_POSITIONS[final_category]
        print(f"--> Sorting {final_category.upper()}.")
        set_servo_angle(target_angle)
        
        # <<< CHANGED: New logic to stop the belt for 'unclassified' items >>>
        if final_category == 'unclassified':
            stop_conveyor() # Ensure conveyor is stopped
            print("\nベルトが止まりました (Belt has stopped).")
            input("[ACTION REQUIRED] Please remove the unclassified item, then press Enter to reset...")
        else:
            # For all other categories, run the conveyor normally
            run_conveyor()
            input("[STEP 4] Belt running. After item is sorted, press Enter to stop...")
        
        stop_conveyor()
        set_servo_angle(SERVO_NEUTRAL_POSITION)
        print("--> System reset. Ready for the next item.")

except KeyboardInterrupt:
    print("\nProgram stopped by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Cleaning up GPIO...")
    GPIO.cleanup()