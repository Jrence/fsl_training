import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand skeletal connection pairs for visual drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ============================================================
# 1. REGISTER CUSTOM LAYER & LOAD MODEL
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

MODEL_PATH = "fsl_model.keras"
CLASSES_PATH = "classes.npy"

model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)
print(f"Loaded {len(classes)} classes successfully!")

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

SEQUENCE_LENGTH = 60
TOTAL_JOINTS = 75
CONFIDENCE_THRESHOLD = 0.75     # 75% Relative Confidence Ratio
MOTION_DELTA_THRESHOLD = 0.0015 # Sensitive hand delta motion

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=4)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Try changing camera index to 1.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nPress 'q' to stop testing.\n")

frame_count = 0
cooldown_counter = 0
missing_hand_counter = 0

pred_label = "Status: Idle (No Hands)"
confidence_text = "Please show hand sign"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = detector_hand.detect(mp_image)
    pose_result = detector_pose.detect(mp_image)

    frame_joints = np.zeros((TOTAL_JOINTS, 3), dtype=np.float32)
    mid_shoulder = np.array([0.5, 0.5, 0.0])
    shoulder_width = 1.0

    # Pose Normalization
    if pose_result.pose_landmarks:
        pose_lms = pose_result.pose_landmarks[0]
        left_shoulder = np.array([pose_lms[11].x, pose_lms[11].y, pose_lms[11].z])
        right_shoulder = np.array([pose_lms[12].x, pose_lms[12].y, pose_lms[12].z])

        mid_shoulder = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width < 1e-6:
            shoulder_width = 1.0

        pose_pts = (np.array([[lm.x, lm.y, lm.z] for lm in pose_lms]) - mid_shoulder) / shoulder_width
        frame_joints[:33] = pose_pts

    # Hand Landmark Routing & Visual Overlay
    if hand_result.hand_landmarks and hand_result.handedness:
        num_hands = len(hand_result.hand_landmarks)

        for hand_lms, hand_info in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = hand_info[0].category_name
            color = (0, 255, 255) if label == 'Right' else (255, 100, 0)

            pts_px = []
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                pts_px.append((cx, cy))
                cv2.circle(frame, (cx, cy), 4, color, -1)

            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, pts_px[start], pts_px[end], color, 2)

        if num_hands >= 2:
            for hand_lms, hand_info in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = hand_info[0].category_name
                hand_pts = (np.array([[lm.x, lm.y, lm.z] for lm in hand_lms]) - mid_shoulder) / shoulder_width
                if label == 'Left':
                    frame_joints[33:54] = hand_pts
                elif label == 'Right':
                    frame_joints[54:75] = hand_pts
        else:
            hand_lms = hand_result.hand_landmarks[0]
            label = hand_result.handedness[0][0].category_name
            hand_pts = (np.array([[lm.x, lm.y, lm.z] for lm in hand_lms]) - mid_shoulder) / shoulder_width
            
            if label == 'Left':
                hand_pts[:, 0] = -hand_pts[:, 0]

            frame_joints[54:75] = hand_pts

    frame_buffer.append(frame_joints)
    frame_count += 1

    if cooldown_counter > 0:
        cooldown_counter -= 1

    # ------------------------------------------------------------
    # STABLE CONTINUOUS PREDICTION LOGIC
    # ------------------------------------------------------------
    if not hand_result.hand_landmarks:
        missing_hand_counter += 1
        # WIPE LANG ANG BUFFER KUNG NAWALA ANG KAMAY NANG HIGIT 10 CONSECUTIVE FRAMES
        if missing_hand_counter > 10:
            pred_label = "Status: Idle (No Hands)"
            confidence_text = "Please show hand sign"
            frame_buffer.clear()
            prediction_history.clear()
            cooldown_counter = 0
    else:
        missing_hand_counter = 0

    if len(frame_buffer) == SEQUENCE_LENGTH and frame_count % 3 == 0:
        buffer_array = np.array(frame_buffer, dtype=np.float32)

        # Delta Motion
        frame_deltas = np.diff(buffer_array[:, 33:75, :], axis=0)
        hand_motion = np.mean(np.abs(frame_deltas))

        if hand_motion < MOTION_DELTA_THRESHOLD:
            if cooldown_counter == 0:
                pred_label = "Status: Idle (Resting)"
                confidence_text = f"Stationary (Delta: {hand_motion:.5f})"
            prediction_history.clear()
            
        elif cooldown_counter == 0:
            predictions = model(np.expand_dims(buffer_array, axis=0), training=False).numpy()[0]
            
            sorted_indices = np.argsort(predictions)[::-1]
            top1_idx, top2_idx = sorted_indices[0], sorted_indices[1]
            top1_prob, top2_prob = predictions[top1_idx], predictions[top2_idx]

            relative_conf = (top1_prob - top2_prob) / (top1_prob + 1e-7)

            if relative_conf >= CONFIDENCE_THRESHOLD:
                current_sign = classes[top1_idx]
                prediction_history.append(current_sign)

                if prediction_history.count(current_sign) >= 2:
                    pred_label = f"Sign: {current_sign}"
                    confidence_text = f"Rel. Conf: {relative_conf * 100:.1f}% (Top 1: {top1_prob * 100:.1f}%)"
                    cooldown_counter = 25
                    prediction_history.clear()
            else:
                pred_label = "Status: Detecting..."
                confidence_text = f"Top: {classes[top1_idx]} (Rel: {relative_conf * 100:.1f}%)"

    # UI Card Overlay
    cv2.rectangle(frame, (15, 15), (460, 95), (20, 20, 20), -1)
    cv2.putText(frame, pred_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, confidence_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    cv2.imshow("FSL Real-Time ST-GCN Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
