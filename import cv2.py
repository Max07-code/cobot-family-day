import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
import sys
import time
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available. Running in simulation mode.")
    GPIO_AVAILABLE = False

class GPIOController:
    """Class to handle GPIO operations for Raspberry Pi"""
    
    def __init__(self, pin=18, active_high=True):
        self.pin = pin
        self.active_high = active_high
        self.gpio_available = GPIO_AVAILABLE
        self.is_setup = False
        
        if self.gpio_available:
            self.setup_gpio()
    
    def setup_gpio(self):
        """Initialize GPIO settings"""
        try:
            GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
            GPIO.setup(self.pin, GPIO.OUT)
            GPIO.output(self.pin, GPIO.LOW if self.active_high else GPIO.HIGH)
            self.is_setup = True
            print(f"GPIO pin {self.pin} initialized successfully")
        except Exception as e:
            print(f"Error setting up GPIO: {e}")
            self.gpio_available = False
    
    def set_pin_high(self):
        """Set the GPIO pin to active state"""
        if self.gpio_available and self.is_setup:
            try:
                GPIO.output(self.pin, GPIO.HIGH if self.active_high else GPIO.LOW)
                print(f"GPIO pin {self.pin} set to HIGH")
            except Exception as e:
                print(f"Error setting pin high: {e}")
        else:
            print("Simulation: GPIO pin would be set to HIGH")
    
    def set_pin_low(self):
        """Set the GPIO pin to inactive state"""
        if self.gpio_available and self.is_setup:
            try:
                GPIO.output(self.pin, GPIO.LOW if self.active_high else GPIO.HIGH)
                print(f"GPIO pin {self.pin} set to LOW")
            except Exception as e:
                print(f"Error setting pin low: {e}")
        else:
            print("Simulation: GPIO pin would be set to LOW")
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.gpio_available and self.is_setup:
            try:
                GPIO.output(self.pin, GPIO.LOW if self.active_high else GPIO.HIGH)
                GPIO.cleanup()
                print("GPIO cleaned up successfully")
            except Exception as e:
                print(f"Error cleaning up GPIO: {e}")

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle between three points.
    Used to determine finger angles.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_thumbs_up(hand_landmarks, img_width, img_height, frame):
    """
    Detects if a "thumbs up" gesture is made based on hand landmarks.
    """
    if not hand_landmarks:
        return False

    # Initialize MediaPipe Hands
    mp_hands_local = mp.solutions.hands

    # Get landmark coordinates (normalized to image size)
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x * img_width, lm.y * img_height])

    # Convert to numpy array for easier indexing
    landmarks = np.array(landmarks)

    # Get important points for thumbs up detection
    wrist = landmarks[mp_hands_local.HandLandmark.WRIST]
    thumb_cmc = landmarks[mp_hands_local.HandLandmark.THUMB_CMC]
    thumb_mcp = landmarks[mp_hands_local.HandLandmark.THUMB_MCP]
    thumb_ip = landmarks[mp_hands_local.HandLandmark.THUMB_IP]
    thumb_tip = landmarks[mp_hands_local.HandLandmark.THUMB_TIP]
    
    # Get indexes for other fingers
    index_mcp = landmarks[mp_hands_local.HandLandmark.INDEX_FINGER_MCP]
    index_pip = landmarks[mp_hands_local.HandLandmark.INDEX_FINGER_PIP]
    index_tip = landmarks[mp_hands_local.HandLandmark.INDEX_FINGER_TIP]
    
    middle_mcp = landmarks[mp_hands_local.HandLandmark.MIDDLE_FINGER_MCP]
    middle_pip = landmarks[mp_hands_local.HandLandmark.MIDDLE_FINGER_PIP]
    middle_tip = landmarks[mp_hands_local.HandLandmark.MIDDLE_FINGER_TIP]
    
    ring_mcp = landmarks[mp_hands_local.HandLandmark.RING_FINGER_MCP]
    ring_pip = landmarks[mp_hands_local.HandLandmark.RING_FINGER_PIP]
    ring_tip = landmarks[mp_hands_local.HandLandmark.RING_FINGER_TIP]
    
    pinky_mcp = landmarks[mp_hands_local.HandLandmark.PINKY_MCP]
    pinky_pip = landmarks[mp_hands_local.HandLandmark.PINKY_PIP]
    pinky_tip = landmarks[mp_hands_local.HandLandmark.PINKY_TIP]

    # Check if thumb is extended
    thumb_direction = thumb_tip - wrist
    
    # Check if other fingers are curled
    index_curl = calculate_angle(index_mcp, index_pip, index_tip)
    middle_curl = calculate_angle(middle_mcp, middle_pip, middle_tip)
    ring_curl = calculate_angle(ring_mcp, ring_pip, ring_tip)
    pinky_curl = calculate_angle(pinky_mcp, pinky_pip, pinky_tip)
    
    # Check if thumb is pointing up
    thumb_pointing_up = thumb_tip[1] < wrist[1] - (img_height * 0.05)
    
    # Check the thumb orientation relative to the palm
    thumb_index_angle = calculate_angle(thumb_tip, wrist, index_mcp)
    
    # RELAXED CRITERIA FOR THUMBS UP
    is_thumbs_up = (
        thumb_pointing_up and                 # Thumb is above wrist
        thumb_index_angle > 25 and            # Thumb is separated from index finger
        index_curl > 70 and                   # Other fingers are somewhat curled
        middle_curl > 70 and
        ring_curl > 70 and
        pinky_curl > 70
    )
    
    if is_thumbs_up:
        # Draw a rectangle around the hand
        x_min = int(np.min(landmarks[:, 0]))
        y_min = int(np.min(landmarks[:, 1]))
        x_max = int(np.max(landmarks[:, 0]))
        y_max = int(np.max(landmarks[:, 1]))
        
        # Add padding to the rectangle
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_width, x_max + padding)
        y_max = min(img_height, y_max + padding)
        
        # Draw the rectangle and text
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, "THUMBS UP!", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (0, 255, 0), 2, cv2.LINE_AA)
        return True

    return False

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Loads, resizes, and normalizes an image."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return image

def load_coco_data(coco_json_path, image_dir, target_size=(224, 224)):
    """
    Loads data from a COCO JSON file for object detection (thumbs up).
    """
    images_data = []
    annotations_data = []

    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

        for annotation in coco_data['annotations']:
            if category_id_to_name[annotation['category_id']] == 'thumbs_up':
                image_id = annotation['image_id']
                image_filename = image_id_to_filename[image_id]
                image_path = os.path.join(image_dir, image_filename)

                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue

                bbox = annotation['bbox']
                image = load_and_preprocess_image(image_path, target_size)
                images_data.append(image)
                annotations_data.append({
                    'image_path': image_path,
                    'bbox': bbox,
                })
        
        return images_data, annotations_data
    
    except Exception as e:
        print(f"Error loading COCO data: {e}")
        return [], []

def create_tensorflow_dataset(images_data, annotations_data):
    """
    Creates a TensorFlow dataset from the loaded image and bounding box data.
    """
    if not images_data or not annotations_data:
        print("Warning: No data to create TensorFlow dataset")
        return None

    def generator():
        for image, annotation in zip(images_data, annotations_data):
            bbox = annotation['bbox']
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height

            img = cv2.imread(annotation['image_path'])
            h, w, _ = img.shape

            x_min_norm = x_min / w
            x_max_norm = x_max / w
            y_min_norm = y_min / h
            y_max_norm = y_max / h
            normalized_bbox = [y_min_norm, x_min_norm, y_max_norm, x_max_norm]
            yield image.astype(np.float32), np.array(normalized_bbox, dtype=np.float32)

    output_types = (tf.float32, tf.float32)
    output_shapes = ((224, 224, 3), (4,))

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=output_types,
        output_shapes=output_shapes
    )
    return dataset

def build_model(input_shape=(224, 224, 3)):
    """Build a simple CNN model for thumbs up detection."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return model

def main():
    # Configuration
    GPIO_PIN = 18  # Change this to your desired GPIO pin
    ACTIVE_HIGH = True  # Set to False if your cobot expects active low signal
    DETECTION_HOLD_TIME = 2.0  # How long to keep the GPIO pin active (seconds)
    DETECTION_COOLDOWN = 1.0  # Minimum time between detections (seconds)
    
    # Initialize GPIO controller
    gpio_controller = GPIOController(pin=GPIO_PIN, active_high=ACTIVE_HIGH)
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        gpio_controller.cleanup()
        return

    print(f"Looking for 'Thumbs Up' gesture. GPIO pin {GPIO_PIN} will be activated.")
    print("Press 'q' to quit.")

    # Training data paths
    base_image_dir = r'C:\Users\maxfe\Desktop\personal projects\computer_vision\train'
    coco_json_path = r'C:\Users\maxfe\Desktop\personal projects\computer_vision\train\_annotations.coco.json'
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thumbs_up_model.h5')
    
    # Load model if available
    model = None
    try:
        if os.path.exists(coco_json_path) and os.path.exists(base_image_dir):
            images_data, annotations_data = load_coco_data(coco_json_path, base_image_dir)
            if images_data and annotations_data:
                train_dataset = create_tensorflow_dataset(images_data, annotations_data)
                if train_dataset:
                    print(f"Successfully loaded {len(images_data)} training images")
                    train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)
                    
                    if os.path.exists(model_save_path):
                        print(f"Loading existing model from {model_save_path}")
                        model = tf.keras.models.load_model(model_save_path)
                    else:
                        print("Building and training a new model...")
                        model = build_model()
                        model.fit(train_dataset, epochs=10)
                        model.save(model_save_path)
                        print(f"Model saved to {model_save_path}")
            else:
                print("No training data found. Using only MediaPipe for detection.")
        else:
            print("Training data paths don't exist. Using only MediaPipe for detection.")
    except Exception as e:
        print(f"Error with training data: {e}")
        print("Using only MediaPipe for detection.")

    # GPIO control variables
    last_detection_time = 0
    gpio_activation_time = 0
    gpio_is_active = False
    
    try:
        # Main detection loop
        with mp_hands.Hands(
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break

                frame = cv2.flip(frame, 1)
                H, W, _ = frame.shape
                current_time = time.time()
                
                # Check if we should turn off the GPIO pin
                if gpio_is_active and (current_time - gpio_activation_time) >= DETECTION_HOLD_TIME:
                    gpio_controller.set_pin_low()
                    gpio_is_active = False
                
                thumbs_up_detected = False
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                # Check MediaPipe results
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )

                        if detect_thumbs_up(hand_landmarks, W, H, frame):
                            thumbs_up_detected = True
                
                # Check with trained model if available
                if model is not None and not thumbs_up_detected:
                    try:
                        input_frame = cv2.resize(rgb_frame, (224, 224))
                        input_frame = input_frame / 255.0
                        input_frame = np.expand_dims(input_frame, axis=0)
                        
                        prediction = model.predict(input_frame, verbose=0)
                        
                        if len(prediction) > 0:
                            y_min, x_min, y_max, x_max = prediction[0]
                            
                            x_min_px = int(x_min * W)
                            y_min_px = int(y_min * H)
                            x_max_px = int(x_max * W)
                            y_max_px = int(y_max * H)
                            
                            box_area = (x_max - x_min) * (y_max - y_min)
                            
                            if box_area > 0.01 and box_area < 0.9:
                                thumbs_up_detected = True
                                cv2.rectangle(frame, (x_min_px, y_min_px), (x_max_px, y_max_px), (0, 255, 255), 2)
                                cv2.putText(frame, "THUMBS UP (Model)", (x_min_px, y_min_px - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Error using model for prediction: {e}")

                # Handle GPIO activation
                if thumbs_up_detected and (current_time - last_detection_time) >= DETECTION_COOLDOWN:
                    if not gpio_is_active:
                        gpio_controller.set_pin_high()
                        gpio_is_active = True
                        gpio_activation_time = current_time
                        last_detection_time = current_time
                        print("Thumbs up detected - Cobot signal activated!")

                # Display status information
                status_text = "Thumbs Up Detected!" if thumbs_up_detected else "No Thumbs Up Detected"
                status_color = (0, 255, 0) if thumbs_up_detected else (255, 0, 0)
                cv2.putText(frame, status_text, (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, status_color, 2, cv2.LINE_AA)
                
                # GPIO status
                gpio_status = f"GPIO Pin {GPIO_PIN}: {'ACTIVE' if gpio_is_active else 'INACTIVE'}"
                gpio_color = (0, 255, 0) if gpio_is_active else (0, 0, 255)
                cv2.putText(frame, gpio_status, (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, gpio_color, 2, cv2.LINE_AA)
                
                # Detection method indicator
                method_text = "Detection: MediaPipe + Model" if model is not None else "Detection: MediaPipe Only"
                cv2.putText(frame, method_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 0), 2, cv2.LINE_AA)

                # Display the frame
                cv2.imshow('Thumbs Up Detector with GPIO Control', frame)

                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up resources
        gpio_controller.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

if __name__ == "__main__":
    main()