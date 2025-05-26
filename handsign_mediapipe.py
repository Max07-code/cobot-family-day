import cv2
import mediapipe as mp
import numpy as np
import time
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Warning: RPi.GPIO not available. Running in simulation mode.")
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
            print(f"GPIO pins {self.pins} initialized successfully")
        except Exception as e:
            print(f"Error setting up GPIO: {e}")
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
                print(f"Number {number} detected - GPIO pin {pin} activated")
            except Exception as e:
                print(f"Error activating pin for number {number}: {e}")
        else:
            print(f"Simulation: Would activate pin for number {number}")
    
    def deactivate_all(self):
        """Deactivate all GPIO pins"""
        if self.gpio_available and self.is_setup:
            try:
                for pin in self.pins:
                    GPIO.output(pin, GPIO.LOW if self.active_high else GPIO.HIGH)
            except Exception as e:
                print(f"Error deactivating pins: {e}")
        else:
            print("Simulation: All pins would be deactivated")
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.gpio_available and self.is_setup:
            try:
                self.deactivate_all()
                GPIO.cleanup()
                print("GPIO cleaned up successfully")
            except Exception as e:
                print(f"Error cleaning up GPIO: {e}")

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
        left_fingers = []
        right_fingers = []
        hand_info = []
        
        # Process each detected hand
        for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
            hand_label = handedness.classification[0].label
            hand_score = handedness.classification[0].score
            
            finger_count, fingers_status = self.count_fingers_on_hand(hand_landmarks, hand_label)
            
            if hand_label == "Left":
                left_hand_count = finger_count
                left_fingers = fingers_status
                hand_info.append(f"Left: {finger_count}")
            else:
                right_hand_count = finger_count
                right_fingers = fingers_status
                hand_info.append(f"Right: {finger_count}")
        
        # Calculate total number
        total_number = left_hand_count + right_hand_count
        
        # Generate confidence based on hand detection quality
        confidence = 0.9 if len(multi_hand_landmarks) == 2 else 0.7  # Higher confidence with both hands
        
        # Create status message
        if len(multi_hand_landmarks) == 1:
            status = f"One hand: {total_number} fingers"
        else:
            status = f"Both hands: L:{left_hand_count} + R:{right_hand_count} = {total_number}"
        
        return total_number, confidence, status
    
    def get_detailed_finger_status(self, multi_hand_landmarks, multi_handedness):
        """Get detailed debug information about finger status"""
        if not multi_hand_landmarks:
            return "No hands detected"
        
        status_lines = []
        
        for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
            hand_label = handedness.classification[0].label
            landmarks = hand_landmarks.landmark
            finger_names = ["thumb", "index", "middle", "ring", "pinky"]
            
            finger_status = []
            for finger in finger_names:
                up = self.is_finger_up(landmarks, finger, hand_label)
                finger_status.append(f"{finger}: {'UP' if up else 'DN'}")
            
            finger_count, _ = self.count_fingers_on_hand(hand_landmarks, hand_label)
            status_lines.append(f"{hand_label} hand ({finger_count}): {' | '.join(finger_status)}")
        
        return " || ".join(status_lines)

def main():
    # Configuration
    GPIO_PINS = [18, 19, 20, 21, 26, 16, 12, 25, 24, 23, 22]  # GPIO pins for numbers 0-10
    ACTIVE_HIGH = True
    CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for detection
    DETECTION_HOLD_TIME = 2.0   # How long to keep GPIO active
    DETECTION_COOLDOWN = 0.5    # Cooldown between detections
    STABILITY_FRAMES = 4        # Frames needed for stable detection
    
    print("Dual Hand Number Detection System (0-10)")
    print("=======================================")
    print("Show numbers 0-10 using your fingers!")
    print("• Use both hands to count higher numbers")
    print("• 0 = No fingers (closed fists)")
    print("• 1-5 = One hand")
    print("• 6-10 = Both hands (e.g., 3 + 3 = 6)")
    print("• Maximum: 5 + 5 = 10")
    print("")
    
    # Initialize GPIO controller
    gpio_controller = GPIOController(pins=GPIO_PINS, active_high=ACTIVE_HIGH)
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize dual hand detector
    detector = DualHandNumberDetector()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        gpio_controller.cleanup()
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera ready. Press 'q' to quit, 'd' to toggle debug mode.")
    
    # Detection variables
    last_detection_time = 0
    gpio_activation_time = 0
    current_active_number = -1
    recent_detections = []
    debug_mode = False
    frame_count = 0
    
    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Detect up to 2 hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as hands:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror the frame
                h, w, _ = frame.shape
                current_time = time.time()
                frame_count += 1
                
                # Check if we should deactivate GPIO
                if current_active_number >= 0 and (current_time - gpio_activation_time) >= DETECTION_HOLD_TIME:
                    gpio_controller.deactivate_all()
                    current_active_number = -1
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True
                
                detected_number = None
                confidence = 0.0
                status_message = ""
                finger_status = ""
                
                # Process hand landmarks
                if results.multi_hand_landmarks and results.multi_handedness:
                    # Draw all detected hands
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )
                    
                    # Detect number using both hands
                    detected_number, confidence, status_message = detector.detect_number_dual_hands(
                        results.multi_hand_landmarks, results.multi_handedness)
                    
                    # Get detailed finger status for debugging
                    if debug_mode:
                        finger_status = detector.get_detailed_finger_status(
                            results.multi_hand_landmarks, results.multi_handedness)
                    
                    # Draw bounding boxes around hands
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        landmarks = hand_landmarks.landmark
                        x_coords = [lm.x * w for lm in landmarks]
                        y_coords = [lm.y * h for lm in landmarks]
                        
                        x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
                        y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20
                        
                        # Different colors for different hands
                        color = (255, 0, 0) if i == 0 else (0, 255, 255)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        
                        # Label the hand
                        hand_label = results.multi_handedness[i].classification[0].label
                        cv2.putText(frame, f"{hand_label} Hand", (x_min, y_min - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    
                    # Display detected number prominently
                    if detected_number is not None and confidence >= CONFIDENCE_THRESHOLD:
                        # Large number display in center
                        cv2.putText(frame, f"{detected_number}", (w//2 - 50, h//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 8, cv2.LINE_AA)
                        
                        # Smaller confidence and status
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", (w//2 - 100, h//2 + 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
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
                
                # Display information on frame
                # Title
                cv2.putText(frame, "Dual Hand Detection (0-10)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Current detection status
                if status_message:
                    cv2.putText(frame, status_message, (10, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0) if detected_number is not None else (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "No hands detected", (10, h - 140), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Stable detection
                stable_text = f"Stable: {stable_number}" if stable_number is not None else "Stable: None"
                cv2.putText(frame, stable_text, (10, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0) if stable_number is not None else (255, 255, 0), 2, cv2.LINE_AA)
                
                # GPIO status
                if current_active_number >= 0:
                    gpio_text = f"GPIO Active: Pin {GPIO_PINS[current_active_number]} (Number {current_active_number})"
                    gpio_color = (0, 255, 0)
                else:
                    gpio_text = "GPIO: Inactive"
                    gpio_color = (0, 0, 255)
                
                cv2.putText(frame, gpio_text, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, gpio_color, 2, cv2.LINE_AA)
                
                # Instructions
                cv2.putText(frame, "Use both hands for 0-10 | 'q' quit | 'd' debug", 
                           (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Hand count indicator
                hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                cv2.putText(frame, f"Hands detected: {hand_count}", (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Debug information
                if debug_mode and finger_status:
                    # Show detailed finger status (may need to wrap text)
                    y_offset = 60
                    for line in finger_status.split(" || "):
                        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.4, (255, 255, 255), 1, cv2.LINE_AA)
                        y_offset += 20
                    
                    # Show recent detections
                    recent_text = f"Recent: {recent_detections}"
                    cv2.putText(frame, recent_text, (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, (128, 128, 128), 1, cv2.LINE_AA)
                
                # Display frame
                try:
                    cv2.imshow('Dual Hand Number Detection (0-10)', frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('d'):
                        debug_mode = not debug_mode
                        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
                        
                except cv2.error as e:
                    # Console mode fallback
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: {status_message}")
                        if debug_mode and finger_status:
                            print(f"Fingers: {finger_status}")
                    time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        gpio_controller.cleanup()
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("Cleanup completed")

if __name__ == "__main__":
    main()