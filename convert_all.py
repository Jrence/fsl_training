import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize Hand & Pose Detectors
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

SOURCE_DIR = "Videos" 
OUTPUT_DIR = "datasets"
SEQUENCE_LENGTH = 60
TOTAL_JOINTS = 75  # 33 Pose + 21 Left Hand + 21 Right Hand

# ============================================================
# PATTERN LIST: Ilista dito ang LAHAT ng folder names na TWO-HANDED
# Kapag nakalista ang folder dito, 100% ng videos sa folder na iyon
# ay gagawing Two-Handed nang walang mintis.
# ============================================================
TWO_HANDED_CLASSES = [
    "Abo",
    "Abril",
    "Agosto",
    "Disyembre",
    "Enero",
    "Hulyo",
    "Hunyo",
    "Kasal na",
    "Kayumanggi",
    "Lila",
    "Linggo",
    "Mabagal",
    "Madilim",
    "Malamig",
    "Marso",
    "Nobyembre",
    "Ngayon",
    "Oktubre",
    "Pebrero",
    "Sabado",
    "Setyembre",
    "Tama"
    
   
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    save_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    video_files = sorted([f for f in os.listdir(class_path) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))])
    
    for video_file in video_files:
        video_prefix = os.path.splitext(video_file)[0]
        output_path = os.path.join(save_class_dir, f"{video_prefix}.npy")
        
        if os.path.exists(output_path):
            print(f"Skipped (May .npy na): {video_file}")
            continue

        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            print(f"Skipping '{video_file}': No frames.")
            continue

        processed_frames_data = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            hand_result = detector_hand.detect(mp_image)
            pose_result = detector_pose.detect(mp_image)
            processed_frames_data.append((hand_result, pose_result))

        # PATTERN DECISION (Folder-Level Consistency)
        if class_name in TWO_HANDED_CLASSES:
            video_has_two_hands = True
        else:
            video_has_two_hands = False

        frame_joints_list = []
        for hand_result, pose_result in processed_frames_data:
            frame_joints = np.zeros((TOTAL_JOINTS, 3))
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
            
            # Hand Mapping Logic
            if hand_result.hand_landmarks and hand_result.handedness:
                if video_has_two_hands:
                    # TWO-HANDED MODE: Hiwalay na Left (33:54) at Right (54:75)
                    for hand_lms, hand_info in zip(hand_result.hand_landmarks, hand_result.handedness):
                        label = hand_info[0].category_name  
                        hand_pts = (np.array([[lm.x, lm.y, lm.z] for lm in hand_lms]) - mid_shoulder) / shoulder_width
                        
                        if label == 'Left':
                            frame_joints[33:54] = hand_pts
                        elif label == 'Right':
                            frame_joints[54:75] = hand_pts
                else:
                    # SINGLE-HANDED MODE: KANANG KAMAY LANG (Laging bagsak sa Right slot 54:75)
                    best_hand_lms = None
                    min_y = float('inf')

                    for hand_lms, hand_info in zip(hand_result.hand_landmarks, hand_result.handedness):
                        avg_y = np.mean([lm.y for lm in hand_lms])
                        if avg_y < min_y:
                            min_y = avg_y
                            best_hand_lms = hand_lms

                    if best_hand_lms is not None:
                        hand_pts = (np.array([[lm.x, lm.y, lm.z] for lm in best_hand_lms]) - mid_shoulder) / shoulder_width
                        frame_joints[54:75] = hand_pts  # deretso sa Right slot

            frame_joints_list.append(frame_joints)

        total_len = len(frame_joints_list)
        if total_len == 0:
            continue

        indices = np.linspace(0, total_len - 1, SEQUENCE_LENGTH).astype(int)
        resampled_sequence = [frame_joints_list[i] for i in indices]

        np.save(output_path, np.array(resampled_sequence, dtype=np.float32))
        print(f"Processed & Saved ({'Two-Handed' if video_has_two_hands else 'Single-Handed Force Right'}): {output_path}")

print("Video dataset conversion complete!")
