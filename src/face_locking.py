import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import pickle
import os
from datetime import datetime
import traceback

# ===================== CONFIGURATION =====================
THRESHOLD = 0.65  # Face recognition similarity threshold
TARGET_NAME = input("Enter the identity to lock onto (e.g., your name): ").strip().lower()
MISS_TOLERANCE = 20  # Frames to tolerate no face before unlock
MOVEMENT_THRESHOLD = 25  # Pixels for left/right movement (reduced for better sensitivity)
BLINK_EAR_THRESHOLD = 0.21  # Lowered for better blink detection
SMILE_CONFIDENCE_THRESHOLD = 0.65  # New smile confidence threshold
CONSECUTIVE_SMILE_FRAMES = 3  # Require smile for N consecutive frames

# ===================== INITIALIZATION =====================
print("Initializing face detection system...")

# Load Haar cascade for face detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize ONNX Runtime session for ArcFace
try:
    model_path = "../models/embedder_arcface.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    session = ort.InferenceSession(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file exists at ../models/embedder_arcface.onnx")
    print("You can download it from: https://github.com/onnx/models/tree/main/vision/body_analysis/arcface")
    exit(1)

# Reference points for face alignment (standard 5-point alignment)
REF_POINTS = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose tip
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]  # Right mouth corner
], dtype=np.float32)

# MediaPipe landmark indices for 5-point alignment
INDICES_5PT = [33, 263, 1, 61, 291]  # Left eye, right eye, nose, left mouth, right mouth

# Eye landmarks for EAR calculation (left and right eyes)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Mouth landmarks for smile detection
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]  # Outer lip contour
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]  # Inner lip contour
MOUTH_TOP = [13, 311, 312, 13, 82]  # Top of upper lip
MOUTH_BOTTOM = [14, 17, 84, 181, 91]  # Bottom of lower lip


# ===================== UTILITY FUNCTIONS =====================
def preprocess(aligned):
    """Preprocess aligned face for ArcFace model"""
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
    img = np.transpose(img, (2, 0, 1))  # Change to CHW format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def get_embedding(aligned):
    """Extract face embedding using ArcFace"""
    blob = preprocess(aligned)
    emb = session.run(None, {'input.1': blob})[0][0]
    return emb / np.linalg.norm(emb)  # Normalize to unit vector


def compute_ear(landmarks, eye_indices, h, w):
    """Compute Eye Aspect Ratio (EAR) for blink detection"""
    points = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                       for i in eye_indices])
    # Calculate distances for EAR formula
    A = np.linalg.norm(points[1] - points[5])  # Vertical distance 1
    B = np.linalg.norm(points[2] - points[4])  # Vertical distance 2
    C = np.linalg.norm(points[0] - points[3])  # Horizontal distance
    return (A + B) / (2.0 * C) if C > 0 else 0


def detect_smile(landmarks, h, w, baseline_mouth_width=None, baseline_lip_sep=None):
    """Advanced smile detection with multiple metrics"""
    # Get key mouth points
    left_mouth = np.array([landmarks.landmark[61].x * w, landmarks.landmark[61].y * h])
    right_mouth = np.array([landmarks.landmark[291].x * w, landmarks.landmark[291].y * h])
    upper_lip_top = np.array([landmarks.landmark[13].x * w, landmarks.landmark[13].y * h])
    lower_lip_bottom = np.array([landmarks.landmark[14].x * w, landmarks.landmark[14].y * h])

    # Calculate metrics
    mouth_width = np.linalg.norm(left_mouth - right_mouth)
    lip_separation = np.linalg.norm(upper_lip_top - lower_lip_bottom)

    # Get reference points for normalization
    left_eye = np.array([landmarks.landmark[33].x * w, landmarks.landmark[33].y * h])
    right_eye = np.array([landmarks.landmark[263].x * w, landmarks.landmark[263].y * h])
    face_width = np.linalg.norm(left_eye - right_eye)

    # Normalize metrics by face width
    normalized_width = mouth_width / face_width if face_width > 0 else 0
    normalized_sep = lip_separation / face_width if face_width > 0 else 0

    # Check mouth corner upward movement (smile lifts corners)
    nose_tip = np.array([landmarks.landmark[1].x * w, landmarks.landmark[1].y * h])
    left_corner_height = left_mouth[1] - nose_tip[1]
    right_corner_height = right_mouth[1] - nose_tip[1]

    # Calculate smile confidence score (0-1)
    smile_score = 0

    # 1. Mouth width expansion (weight: 0.4)
    if normalized_width > 0.35:  # Typical smiling mouth is wider
        width_score = min((normalized_width - 0.35) * 10, 1.0)
        smile_score += width_score * 0.4

    # 2. Lip separation (teeth showing) (weight: 0.3)
    if normalized_sep > 0.08:  # Smiles usually show teeth
        sep_score = min((normalized_sep - 0.08) * 20, 1.0)
        smile_score += sep_score * 0.3

    # 3. Mouth corner upward movement (weight: 0.3)
    corner_up_score = 0
    if left_corner_height < -5 and right_corner_height < -5:  # Both corners above nose
        corner_up_score = 1.0
    elif left_corner_height < 0 or right_corner_height < 0:  # At least one above nose
        corner_up_score = 0.6
    smile_score += corner_up_score * 0.3

    # Use baseline comparison if available
    if baseline_mouth_width and baseline_lip_sep:
        width_increase = mouth_width / baseline_mouth_width if baseline_mouth_width > 0 else 1
        sep_increase = lip_separation / baseline_lip_sep if baseline_lip_sep > 0 else 1

        # Bonus points for relative increase
        if width_increase > 1.15:  # 15% wider than baseline
            smile_score += min(width_increase - 1, 0.2)
        if sep_increase > 1.3:  # 30% more separation than baseline
            smile_score += min(sep_increase - 1, 0.2)

    # Clamp to [0, 1]
    smile_score = min(max(smile_score, 0), 1)

    return smile_score > SMILE_CONFIDENCE_THRESHOLD, smile_score, normalized_width, normalized_sep


class ActionDetector:
    """Class to manage action detection with state tracking"""

    def __init__(self):
        self.prev_nose_x = None
        self.prev_ear = None
        self.baseline_mouth_width = None
        self.baseline_lip_sep = None
        self.smile_frames = 0
        self.blink_frames = 0
        self.action_cooldown = 0

    def update_baseline(self, landmarks, h, w):
        """Update neutral face baseline metrics"""
        left_mouth = np.array([landmarks.landmark[61].x * w, landmarks.landmark[61].y * h])
        right_mouth = np.array([landmarks.landmark[291].x * w, landmarks.landmark[291].y * h])
        upper_lip_top = np.array([landmarks.landmark[13].x * w, landmarks.landmark[13].y * h])
        lower_lip_bottom = np.array([landmarks.landmark[14].x * w, landmarks.landmark[14].y * h])

        self.baseline_mouth_width = np.linalg.norm(left_mouth - right_mouth)
        self.baseline_lip_sep = np.linalg.norm(upper_lip_top - lower_lip_bottom)

    def detect_actions(self, landmarks, h, w, x_offset, y_offset, locked):
        """Detect all facial actions"""
        actions = []

        if not locked:
            return actions

        # Movement detection
        nose_x = int(landmarks.landmark[1].x * w) + x_offset
        if self.prev_nose_x is not None:
            delta_x = nose_x - self.prev_nose_x
            if abs(delta_x) > MOVEMENT_THRESHOLD:
                direction = "right" if delta_x > 0 else "left"
                actions.append(f"moved {direction} ({abs(delta_x):.0f}px)")

        self.prev_nose_x = nose_x

        # Blink detection
        ear_left = compute_ear(landmarks, LEFT_EYE, h, w)
        ear_right = compute_ear(landmarks, RIGHT_EYE, h, w)
        ear = (ear_left + ear_right) / 2

        if ear < BLINK_EAR_THRESHOLD:
            self.blink_frames += 1
            if self.blink_frames == 2:  # Confirm blink on 2nd frame
                actions.append(f"blink (EAR: {ear:.2f})")
        else:
            self.blink_frames = 0

        # Smile detection
        is_smiling, smile_score, mouth_width_norm, lip_sep_norm = detect_smile(
            landmarks, h, w, self.baseline_mouth_width, self.baseline_lip_sep
        )

        if is_smiling:
            self.smile_frames += 1
            if self.smile_frames >= CONSECUTIVE_SMILE_FRAMES:
                actions.append(f"smile (score: {smile_score:.2f}, width: {mouth_width_norm:.3f})")
        else:
            self.smile_frames = 0
            # Update baseline when not smiling and neutral
            if 0.25 < mouth_width_norm < 0.33 and 0.05 < lip_sep_norm < 0.08:
                self.update_baseline(landmarks, h, w)

        # Apply cooldown to avoid duplicate actions
        if self.action_cooldown > 0:
            self.action_cooldown -= 1

        return actions


# ===================== LOAD FACE DATABASE =====================
try:
    db_path = '../data/db/face_db.pkl'
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    with open(db_path, 'rb') as f:
        db = pickle.load(f)

    # Process embeddings
    reference = {}
    for name, embs in db.items():
        if len(embs) > 0:
            mean_emb = np.mean(np.array(embs), axis=0)
            mean_emb /= np.linalg.norm(mean_emb)
            reference[name.lower()] = mean_emb

    if TARGET_NAME not in reference:
        print(f"Error: {TARGET_NAME} not found in database!")
        print(f"Available identities: {list(reference.keys())}")
        exit(1)

    target_emb = reference[TARGET_NAME]
    print(f"Loaded database with {len(reference)} identities")

except Exception as e:
    print(f"Error loading database: {e}")
    traceback.print_exc()
    exit(1)

# ===================== MAIN LOOP =====================
action_detector = ActionDetector()
locked = False
miss_count = 0
prev_bbox = None
history_file = None
locked_timestamp = None
fps_counter = 0
start_time = datetime.now()

print("\n" + "=" * 50)
print(f"Target: {TARGET_NAME}")
print("Controls:")
print("  - Press 'q' to quit")
print("  - Make sure face is well-lit and visible")
print("=" * 50 + "\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Flip horizontally for mirror view and get frame dimensions
    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]
    fps_counter += 1

    # Determine region of interest
    if locked and prev_bbox:
        x, y, w, h = prev_bbox
        margin = 120  # Increased margin for better tracking
        roi_x = max(0, x - margin)
        roi_y = max(0, y - margin)
        roi_w = min(w_frame - roi_x, w + 2 * margin)
        roi_h = min(h_frame - roi_y, h + 2 * margin)
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        offset_x, offset_y = roi_x, roi_y
    else:
        roi = frame
        offset_x, offset_y = 0, 0

    # Convert to grayscale for Haar cascade
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    face_found = len(faces) > 0

    if face_found:
        # Take the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # Convert to full frame coordinates
        x += offset_x
        y += offset_y

        # Extract face region
        crop = frame[y:y + h, x:x + w]

        # Ensure crop is valid
        if crop.size == 0:
            continue

        # Convert to RGB for MediaPipe
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_crop)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]

            # Face alignment using 5-point landmarks
            pts = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h]
                            for i in INDICES_5PT], dtype=np.float32)

            try:
                M, _ = cv2.estimateAffinePartial2D(pts, REF_POINTS)
                aligned = cv2.warpAffine(crop, M, (112, 112), flags=cv2.INTER_LINEAR)

                # Face recognition
                query_emb = get_embedding(aligned)
                similarity = np.dot(query_emb, target_emb)

                # Action detection
                actions = action_detector.detect_actions(lm, h, w, x, y, locked)

                # Locking logic
                if not locked and similarity >= THRESHOLD:
                    locked = True
                    miss_count = 0
                    locked_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    history_file = f"../data/{TARGET_NAME}_history_{locked_timestamp}.txt"

                    # Initialize baseline for action detection
                    action_detector.update_baseline(lm, h, w)

                    with open(history_file, 'w') as f:
                        f.write(f"Face locking started for {TARGET_NAME} at {datetime.now()}\n")
                        f.write(f"Initial similarity: {similarity:.4f}\n")
                        f.write("-" * 50 + "\n")

                    print(f"\n✓ LOCKED onto {TARGET_NAME} (similarity: {similarity:.3f})")
                    print(f"  History saved to: {history_file}")

                # Log actions to history file
                if locked and actions and history_file:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    with open(history_file, 'a') as f:
                        for action in actions:
                            f.write(f"{timestamp} | {action}\n")

                # Draw bounding box and info
                if locked:
                    color = (0, 255, 0)  # Green for locked
                    status_text = f"LOCKED: {TARGET_NAME} ({similarity:.3f})"

                    # Draw thicker box for locked state
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(frame, status_text, (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # Display actions
                    for i, action in enumerate(actions[:3]):  # Show up to 3 actions
                        cv2.putText(frame, f"• {action}", (x, y + h + 30 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                else:
                    color = (255, 0, 0)  # Blue for searching
                    status_text = f"Searching: {TARGET_NAME} ({similarity:.3f})"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, status_text, (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Display aligned face
                cv2.imshow('Aligned Face', aligned)

                # Update tracking variables
                prev_bbox = (x, y, w, h)
                miss_count = 0

            except Exception as e:
                print(f"Error in face processing: {e}")
                continue
        else:
            face_found = False
    else:
        # No face detected
        if locked:
            miss_count += 1
            if miss_count > MISS_TOLERANCE:
                locked = False
                print("\n⚠ Lock released - face disappeared")
                if history_file:
                    with open(history_file, 'a') as f:
                        f.write(f"\nLock released at {datetime.now()}\n")
                        f.write(f"Total tracking duration: {datetime.now() - start_time}\n")
                history_file = None
                action_detector = ActionDetector()  # Reset action detector

    # Display FPS and status
    elapsed = (datetime.now() - start_time).total_seconds()
    fps = fps_counter / elapsed if elapsed > 0 else 0

    status_line = f"FPS: {fps:.1f} | "
    status_line += f"Status: {'LOCKED' if locked else 'SEARCHING'} | "
    status_line += f"Faces: {len(faces)}"

    cv2.putText(frame, status_line, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display frame
    cv2.imshow('Face Locking System', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nExiting...")
        break
    elif key == ord('r') and locked:
        print("\nManual lock release")
        locked = False
        if history_file:
            with open(history_file, 'a') as f:
                f.write(f"\nManual lock release at {datetime.now()}\n")

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()

# Print summary
if locked_timestamp:
    print(f"\nSession summary:")
    print(f"  Target: {TARGET_NAME}")
    print(f"  Lock time: {locked_timestamp}")
    if history_file and os.path.exists(history_file):
        print(f"  History file: {history_file}")
        with open(history_file, 'r') as f:
            lines = f.readlines()
            action_count = len([l for l in lines if '|' in l])
            print(f"  Actions logged: {action_count}")
    print(f"  Total runtime: {datetime.now() - start_time}")