import cv2
import mediapipe as mp
import numpy as np
import time
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

class GPIOController:
    """Class to handle GPIO operations for Raspberry Pi"""
    
    def __init__(self, pins=None, active_high=True):
        # Default pins for numbers 0-10 (need 11 pins total)
        self.pins = pins if pins else [18, 19, 20, 21, 26, 16, 12, 25, 24, 23, 22]  # GPIO pins for numbers 0-10
        self.active_high = active_high
        self.gpio_available = GPIO_AVAILABLE
        self.is_setup = False
        
        if self.gpio_available:
            self.setup_gpio()
    
    def setup_gpio(self):
        """Initialize GPIO settings"""
        try:
            GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
            for pin in self.pins:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW if self.active_high else GPIO.HIGH)
            self.is_setup = True
        except Exception as e:
            self.gpio_available = False
    
    def activate_number(self, number):
        """Activate the GPIO pin corresponding to the detected number (0-10)"""
        if 0 <= number <= 10 and self.gpio_available and self.is_setup:
            try:
                # First, turn off all pins
                self.deactivate_all()
                # Then activate the specific pin for this number
                pin = self.pins[number]
                GPIO.output(pin, GPIO.HIGH if self.active_high else GPIO.LOW)
            except Exception as e:
                pass
    
    def deactivate_all(self):
        """Deactivate all GPIO pins"""
        if self.gpio_available and self.is_setup:
            try:
                for pin in self.pins:
                    GPIO.output(pin, GPIO.LOW if self.active_high else GPIO.HIGH)
            except Exception as e:
                pass
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.gpio_available and self.is_setup:
            try:
                self.deactivate_all()
                GPIO.cleanup()
            except Exception as e:
                pass

class DualHandNumberDetector:
    """Detect numbers 0-10 using both hands with MediaPipe hand landmarks"""
    
    def __init__(self):
        # MediaPipe hand landmark indices
        self.THUMB_TIP = 4
        self.THUMB_IP = 3
        
        self.INDEX_TIP = 8
        self.INDEX_PIP = 6
        
        self.MIDDLE_TIP = 12
        self.MIDDLE_PIP = 10
        
        self.RING_TIP = 16
        self.RING_PIP = 14
        
        self.PINKY_TIP = 20
        self.PINKY_PIP = 18
    
    def is_finger_up(self, landmarks, finger_name, hand_label):
        """Check if a specific finger is extended (up)"""
        if finger_name == "thumb":
            # For thumb, check based on hand orientation
            if hand_label == "Left":  # Left hand (appears on right side of screen)
                return landmarks[self.THUMB_TIP].x < landmarks[self.THUMB_IP].x
            else:  # Right hand (appears on left side of screen)
                return landmarks[self.THUMB_TIP].x > landmarks[self.THUMB_IP].x
        
        elif finger_name == "index":
            return landmarks[self.INDEX_TIP].y < landmarks[self.INDEX_PIP].y
        
        elif finger_name == "middle":
            return landmarks[self.MIDDLE_TIP].y < landmarks[self.MIDDLE_PIP].y
        
        elif finger_name == "ring":
            return landmarks[self.RING_TIP].y < landmarks[self.RING_PIP].y
        
        elif finger_name == "pinky":
            return landmarks[self.PINKY_TIP].y < landmarks[self.PINKY_PIP].y
        
        return False
    
    def count_fingers_on_hand(self, hand_landmarks, hand_label):
        """Count fingers on a single hand"""
        if not hand_landmarks:
            return 0, []
        
        landmarks = hand_landmarks.landmark
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        fingers_up = []
        
        for finger in finger_names:
            fingers_up.append(self.is_finger_up(landmarks, finger, hand_label))
        
        return sum(fingers_up), fingers_up
    
    def detect_number_dual_hands(self, multi_hand_landmarks, multi_handedness):
        """Detect number (0-10) based on finger positions from both hands"""
        if not multi_hand_landmarks:
            return None, 0.0, "No hands detected"
        
        left_hand_count = 0
        right_hand_count = 0
        
        # Process each detected hand
        for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
            hand_label = handedness.classification[0].label
            finger_count, _ = self.count_fingers_on_hand(hand_landmarks, hand_label)
            
            if hand_label == "Left":
                left_hand_count = finger_count
            else:
                right_hand_count = finger_count
        
        # Calculate total number
        total_number = left_hand_count + right_hand_count
        
        # Generate confidence based on hand detection quality
        confidence = 0.9 if len(multi_hand_landmarks) == 2 else 0.7  # Higher confidence with both hands
        
        return total_number, confidence, ""

def main():
    # Configuration
    GPIO_PINS = [18, 19, 20, 21, 26, 16, 12, 25, 24, 23, 22]  # GPIO pins for numbers 0-10
    ACTIVE_HIGH = True
    CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for detection
    DETECTION_HOLD_TIME = 2.0   # How long to keep GPIO active
    DETECTION_COOLDOWN = 0.5    # Cooldown between detections
    STABILITY_FRAMES = 4        # Frames needed for stable detection
    
    # Initialize GPIO controller
    gpio_controller = GPIOController(pins=GPIO_PINS, active_high=ACTIVE_HIGH)
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    
    # Initialize dual hand detector
    detector = DualHandNumberDetector()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        gpio_controller.cleanup()
        return
    
    # Set camera properties for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced for speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced for speed
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
    
    # Detection variables
    last_detection_time = 0
    gpio_activation_time = 0
    current_active_number = -1
    recent_detections = []
    frame_count = 0
    detected_number = None
    confidence = 0.0
    
    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Detect up to 2 hands
            min_detection_confidence=0.6,  # Lowered for speed
            min_tracking_confidence=0.4,   # Lowered for speed
            model_complexity=0) as hands:   # Fastest model
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)  # Mirror the frame
                current_time = time.time()
                frame_count += 1
                
                # Check if we should deactivate GPIO
                if current_active_number >= 0 and (current_time - gpio_activation_time) >= DETECTION_HOLD_TIME:
                    gpio_controller.deactivate_all()
                    current_active_number = -1
                
                # Process every 2nd frame for speed
                if frame_count % 2 == 0:
                    # Convert BGR to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False
                    results = hands.process(rgb_frame)
                    
                    # Process hand landmarks
                    if results.multi_hand_landmarks and results.multi_handedness:
                        # Detect number using both hands
                        detected_number, confidence, _ = detector.detect_number_dual_hands(
                            results.multi_hand_landmarks, results.multi_handedness)
                    else:
                        detected_number = None
                        confidence = 0.0
                    
                    # Stability checking
                    if detected_number is not None and confidence >= CONFIDENCE_THRESHOLD:
                        recent_detections.append(detected_number)
                    else:
                        recent_detections.append(-1)
                    
                    # Keep only recent frames
                    if len(recent_detections) > STABILITY_FRAMES:
                        recent_detections.pop(0)
                    
                    # Check for stable detection
                    stable_number = None
                    if len(recent_detections) >= STABILITY_FRAMES:
                        # Check if majority of recent detections are the same
                        last_detection = recent_detections[-1]
                        if last_detection != -1:
                            count = recent_detections.count(last_detection)
                            if count >= (STABILITY_FRAMES * 0.75):  # 75% agreement
                                stable_number = last_detection
                    
                    # Handle GPIO activation
                    if (stable_number is not None and 
                        (current_time - last_detection_time) >= DETECTION_COOLDOWN and
                        stable_number != current_active_number):
                        
                        gpio_controller.activate_number(stable_number)
                        current_active_number = stable_number
                        gpio_activation_time = current_time
                        last_detection_time = current_time
                
                # Scale frame for display while keeping processing resolution
                display_frame = cv2.resize(frame, (1280, 960))  # Much bigger display
                
                # Minimal visual feedback (comment out for headless)
                if detected_number is not None and confidence >= CONFIDENCE_THRESHOLD:
                    cv2.putText(display_frame, f"{detected_number}", (500, 500), 
                               cv2.FONT_HERSHEY_SIMPLEX, 15, (0, 255, 0), 20)
                
                cv2.putText(display_frame, f"GPIO: {current_active_number if current_active_number >= 0 else 'OFF'}", 
                           (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8)
                
                # Display frame (comment out for headless production)
                try:
                    cv2.imshow('Hand Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    # Headless mode - no display
                    pass
                
                # Small delay to prevent CPU overload
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        # Cleanup
        gpio_controller.cleanup()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()