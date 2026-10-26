import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time

# Hand skeletal connection pairs for visual drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

SEQUENCE_LENGTH = 60
TOTAL_JOINTS = 75
CLASSES_PATH = "classes.npy"
TFLITE_PATH = "fsl_model.tflite"
KERAS_PATH = "fsl_model.keras"

# ============================================================
# 1. REGISTER CUSTOM LAYER & LOAD MODEL (TFLITE OR KERAS)
# ============================================================
@tf.keras.utils.register_keras_serializable()
class TFLiteGraphConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, adj_matrix, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.adj_matrix = adj_matrix
        self.A = tf.constant(adj_matrix, dtype=tf.float32)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.out_channels, use_bias=False)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_channels": self.out_channels,
            "adj_matrix": self.adj_matrix.tolist() if isinstance(self.adj_matrix, np.ndarray) else self.adj_matrix
        })
        return config

    def call(self, inputs):
        x_weight = self.dense(inputs)
        shape = tf.shape(inputs)
        batch_size, frames = shape[0], shape[1]
        x_reshaped = tf.reshape(x_weight, [-1, 75, self.out_channels])
        x_graph = tf.matmul(self.A, x_reshaped)
        return tf.reshape(x_graph, [batch_size, frames, 75, self.out_channels])

if not os.path.exists(CLASSES_PATH):
    print(f"\n[ERROR] '{CLASSES_PATH}' not found!")
    print(f"Please train the model first by running 'python train.py' to generate classes and weights.\n")
    exit(1)

classes = np.load(CLASSES_PATH)
print(f"Loaded {len(classes)} classes from {CLASSES_PATH}")

use_tflite = os.path.exists(TFLITE_PATH)
if use_tflite:
    print(f"Initializing accelerated TFLite engine: {TFLITE_PATH}")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def predict_sequence(seq_tensor):
        interpreter.set_tensor(input_details[0]['index'], seq_tensor.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]
elif os.path.exists(KERAS_PATH):
    print(f"TFLite model not found. Using Keras fallback: {KERAS_PATH}")
    keras_model = tf.keras.models.load_model(KERAS_PATH, custom_objects={'TFLiteGraphConv': TFLiteGraphConv})

    def predict_sequence(seq_tensor):
        return keras_model(seq_tensor, training=False).numpy()[0]
else:
    print(f"\n[ERROR] Model file not found!")
    print(f"Neither '{TFLITE_PATH}' nor '{KERAS_PATH}' exists in the current directory.")
    print(f"Please follow the workflow:")
    print(f"  1. python convert_all.py  (Extract features from Videos)")
    print(f"  2. python train.py        (Train the ST-GCN network)")
    print(f"  3. python convert_to_tflite.py (Convert for ultra-fast inference)")
    print(f"  4. python predict.py      (Run live webcam detection)\n")
    exit(1)

# ============================================================
# 2. MEDIAPIPE DETECTORS & SETTINGS
# ============================================================
base_hand = python.BaseOptions(model_asset_path='hand_landmarker.task')
options_hand = vision.HandLandmarkerOptions(
    base_options=base_hand, 
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE
)
detector_hand = vision.HandLandmarker.create_from_options(options_hand)

base_pose = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options_pose = vision.PoseLandmarkerOptions(
    base_options=base_pose,
    running_mode=vision.RunningMode.IMAGE
)
detector_pose = vision.PoseLandmarker.create_from_options(options_pose)

# ============================================================
# 3. WEBCAM INITIALIZATION WITH LOW LATENCY SETTINGS
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Disable camera internal frame buffering to eliminate queue lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 60)

print("\nReal-Time FSL Detection Running. Press 'q' to quit.\n")

# Normalization caching to prevent coordinate jumps
cached_mid_shoulder = np.array([0.5, 0.5, 0.0], dtype=np.float32)
cached_shoulder_width = 0.20
cached_left_wrist = np.array([0.4, 0.7], dtype=np.float32)
cached_right_wrist = np.array([0.6, 0.7], dtype=np.float32)

# Anti-occlusion landmark smoothing for sideways hands
smoothed_left_hand = None
smoothed_right_hand = None
ALPHA_SMOOTH = 0.75  # 0.75 = responsive with anti-jitter filter

# Rolling buffer for real-time continuous inference
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
prob_smoothing_history = deque(maxlen=4)
missing_hand_counter = 0
cooldown_counter = 0

prev_hand_pts = None
frame_idx = 0
fps_history = deque(maxlen=30)
prev_time = time.time()
CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence to display prediction (random chance is 0.78%)

pred_label = "Status: Ready"
confidence_text = "Perform gesture or show hand"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    curr_time = time.time()
    fps = 1.0 / max(1e-5, (curr_time - prev_time))
    prev_time = curr_time
    fps_history.append(fps)
    avg_fps = sum(fps_history) / len(fps_history)

    # Flip horizontally for natural mirror display
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Downscale frame for fast MediaPipe landmark extraction (2x speedup)
    small_frame = cv2.resize(frame, (320, 240))
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)

    # Run Hand Landmarker on every frame
    hand_result = detector_hand.detect(mp_image)

    # Run Pose Landmarker once every 4 frames (saves ~35% CPU overhead)
    if frame_idx % 4 == 0 or cached_shoulder_width == 0.20:
        pose_result = detector_pose.detect(mp_image)
        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]
            left_s = np.array([pose_lms[11].x, pose_lms[11].y, pose_lms[11].z])
            right_s = np.array([pose_lms[12].x, pose_lms[12].y, pose_lms[12].z])
            s_width = np.linalg.norm(left_s - right_s)
            if s_width > 0.05:
                cached_mid_shoulder = (left_s + right_s) / 2.0
                cached_shoulder_width = s_width
            # Cache pose body wrists (Joint 15 = Left wrist, Joint 16 = Right wrist)
            cached_left_wrist = np.array([pose_lms[15].x, pose_lms[15].y], dtype=np.float32)
            cached_right_wrist = np.array([pose_lms[16].x, pose_lms[16].y], dtype=np.float32)

    frame_joints = np.zeros((TOTAL_JOINTS, 3), dtype=np.float32)

    # Pose slot (0:33)
    if 'pose_result' in locals() and pose_result.pose_landmarks:
        pose_lms = pose_result.pose_landmarks[0]
        pose_pts = (np.array([[lm.x, lm.y, lm.z] for lm in pose_lms]) - cached_mid_shoulder) / cached_shoulder_width
        frame_joints[:33] = pose_pts

    # Hand Landmark Routing & Visual Overlay
    current_hand_motion = 0.0
    has_hand = False

    if hand_result.hand_landmarks and hand_result.handedness:
        has_hand = True
        num_hands = len(hand_result.hand_landmarks)

        # Draw overlays on display frame
        for hand_lms, hand_info in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = hand_info[0].category_name
            color = (0, 255, 255) if label == 'Right' else (255, 100, 0)
            pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for cx, cy in pts_px:
                cv2.circle(frame, (cx, cy), 3, color, -1)
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, pts_px[start], pts_px[end], color, 2)

        # Slot assignment with Body-Wrist Proximity Matching (Solves overlapping palms)
        if num_hands >= 2:
            hand_0 = hand_result.hand_landmarks[0]
            hand_1 = hand_result.hand_landmarks[1]

            w0 = np.array([hand_0[0].x, hand_0[0].y], dtype=np.float32)
            w1 = np.array([hand_1[0].x, hand_1[0].y], dtype=np.float32)

            pts_0 = (np.array([[lm.x, lm.y, lm.z] for lm in hand_0]) - cached_mid_shoulder) / cached_shoulder_width
            pts_1 = (np.array([[lm.x, lm.y, lm.z] for lm in hand_1]) - cached_mid_shoulder) / cached_shoulder_width

            # Distance cost matching: match hands to anatomical arms
            cost_direct = np.linalg.norm(w0 - cached_left_wrist) + np.linalg.norm(w1 - cached_right_wrist)
            cost_crossed = np.linalg.norm(w0 - cached_right_wrist) + np.linalg.norm(w1 - cached_left_wrist)

            if cost_direct <= cost_crossed:
                raw_left, raw_right = pts_0, pts_1
            else:
                raw_left, raw_right = pts_1, pts_0

            # Anti-occlusion temporal smoothing (Solves sideways hands & flicker)
            if smoothed_left_hand is None:
                smoothed_left_hand = raw_left.copy()
            else:
                smoothed_left_hand = ALPHA_SMOOTH * raw_left + (1.0 - ALPHA_SMOOTH) * smoothed_left_hand

            if smoothed_right_hand is None:
                smoothed_right_hand = raw_right.copy()
            else:
                smoothed_right_hand = ALPHA_SMOOTH * raw_right + (1.0 - ALPHA_SMOOTH) * smoothed_right_hand

            frame_joints[33:54] = smoothed_left_hand
            frame_joints[54:75] = smoothed_right_hand
        else:
            # Single-Handed signs: Always slot 54:75 (Right slot)
            hand_lms = hand_result.hand_landmarks[0]
            pts = (np.array([[lm.x, lm.y, lm.z] for lm in hand_lms]) - cached_mid_shoulder) / cached_shoulder_width

            # Anti-occlusion temporal smoothing for sideways fists
            if smoothed_right_hand is None:
                smoothed_right_hand = pts.copy()
            else:
                smoothed_right_hand = ALPHA_SMOOTH * pts + (1.0 - ALPHA_SMOOTH) * smoothed_right_hand

            smoothed_left_hand = None
            frame_joints[54:75] = smoothed_right_hand

        # Compute instant hand motion delta
        active_pts = frame_joints[54:75]
        if prev_hand_pts is not None:
            current_hand_motion = np.mean(np.linalg.norm(active_pts - prev_hand_pts, axis=1))
        prev_hand_pts = active_pts.copy()
    else:
        prev_hand_pts = None
        smoothed_left_hand = None
        smoothed_right_hand = None

    # ============================================================
    # 4. CONTINUOUS ULTRA-RESPONSIVE INFERENCE
    # ============================================================
    if has_hand:
        frame_buffer.append(frame_joints)
        missing_hand_counter = 0

        # Predict every 3 frames once at least 10 frames of hand are observed
        if len(frame_buffer) >= 10 and frame_idx % 3 == 0:
            # Resample current buffer to exactly 60 frames for instant prediction
            buf_len = len(frame_buffer)
            idx = np.linspace(0, buf_len - 1, SEQUENCE_LENGTH).astype(int)
            seq_array = np.array([frame_buffer[i] for i in idx], dtype=np.float32)

            preds = predict_sequence(np.expand_dims(seq_array, axis=0))
            prob_smoothing_history.append(preds)

            # Smooth probabilities over last few predictions
            smoothed = np.mean(prob_smoothing_history, axis=0)
            sorted_idx = np.argsort(smoothed)[::-1]
            top1, top2 = sorted_idx[0], sorted_idx[1]
            conf1, conf2 = smoothed[top1], smoothed[top2]

            name1 = classes[top1] if len(classes) > top1 else f"Class {top1}"
            name2 = classes[top2] if len(classes) > top2 else f"Class {top2}"

            if conf1 >= CONFIDENCE_THRESHOLD:
                pred_label = f"Sign: {name1}"
                confidence_text = f"Top 1: {conf1 * 100:.1f}% | 2nd: {name2} ({conf2 * 100:.1f}%)"
            else:
                pred_label = f"Detecting: {name1}?"
                confidence_text = f"Top 1: {conf1 * 100:.1f}% | 2nd: {name2} ({conf2 * 100:.1f}%)"
    else:
        missing_hand_counter += 1
        if missing_hand_counter > 12:
            frame_buffer.clear()
            prob_smoothing_history.clear()
            pred_label = "Status: Idle (No Hands)"
            confidence_text = "Show your hand to the camera"

    # UI Information Card Overlay
    cv2.rectangle(frame, (15, 15), (510, 95), (20, 20, 20), -1)
    cv2.putText(frame, pred_label, (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(frame, confidence_text, (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

    # FPS Indicator
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (w - 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    cv2.imshow("FSL Real-Time High-Speed Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
