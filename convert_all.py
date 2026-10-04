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
# PATTERN LIST: ALL TWO-HANDED CLASS FOLDERS
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

if not os.path.isdir(SOURCE_DIR):
    raise FileNotFoundError(
        f"Source folder not found: {SOURCE_DIR}"
    )

# ============================================================
# PROCESS VIDEO FOLDERS
# ============================================================
for class_name in sorted(os.listdir(SOURCE_DIR)):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    save_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    video_files = sorted([
        f for f in os.listdir(class_path)
        if f.lower().endswith(
            ('.mp4', '.mov', '.avi', '.mkv')
        )
    ])

    for video_file in video_files:
        video_prefix = os.path.splitext(video_file)[0]
        output_path = os.path.join(
            save_class_dir,
            f"{video_prefix}.npy"
        )

        # Skip existing NPY files
        if os.path.exists(output_path):
            print(f"Skipped (May .npy na): {video_file}")
            continue

        video_path = os.path.join(class_path, video_file)

        print("\n" + "=" * 60)
        print(f"Processing: {video_path}")
        print("=" * 60)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {video_file}")
            continue

        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            print(f"Skipping '{video_file}': No frames.")
            continue

        # Folder-level hand configuration
        video_has_two_hands = (
            class_name in TWO_HANDED_CLASSES
        )

        frame_joints_list = []

        # Keep latest valid pose normalization reference
        previous_mid_shoulder = np.array(
            [0.5, 0.5, 0.0], dtype=np.float32
        )
        previous_shoulder_width = 1.0

        hand_detected_frames = 0
        two_hand_detected_frames = 0
        pose_detected_frames = 0

        # ====================================================
        # PROCESS EACH FRAME
        # ====================================================
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            try:
                hand_result = detector_hand.detect(mp_image)
                pose_result = detector_pose.detect(mp_image)
            except Exception as e:
                print(f"Frame detection warning: {e}")
                frame_joints_list.append(
                    np.zeros((TOTAL_JOINTS, 3), dtype=np.float32)
                )
                continue

            frame_joints = np.zeros(
                (TOTAL_JOINTS, 3), dtype=np.float32
            )

            mid_shoulder = previous_mid_shoulder.copy()
            shoulder_width = previous_shoulder_width

            # ------------------------------------------------
            # POSE NORMALIZATION
            # ------------------------------------------------
            if pose_result.pose_landmarks:
                pose_lms = pose_result.pose_landmarks[0]

                pose_array = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in pose_lms
                ], dtype=np.float32)

                left_shoulder = pose_array[11]
                right_shoulder = pose_array[12]

                detected_mid_shoulder = (
                    left_shoulder + right_shoulder
                ) / 2.0

                detected_shoulder_width = np.linalg.norm(
                    left_shoulder - right_shoulder
                )

                if (
                    np.isfinite(detected_shoulder_width)
                    and detected_shoulder_width > 1e-6
                ):
                    mid_shoulder = detected_mid_shoulder
                    shoulder_width = detected_shoulder_width

                    previous_mid_shoulder = mid_shoulder.copy()
                    previous_shoulder_width = shoulder_width

                pose_pts = (
                    pose_array - mid_shoulder
                ) / shoulder_width

                if np.all(np.isfinite(pose_pts)):
                    frame_joints[:33] = pose_pts
                    pose_detected_frames += 1

            # ------------------------------------------------
            # HAND MAPPING
            # ------------------------------------------------
            detected_hands = (
                hand_result.hand_landmarks
                if hand_result is not None
                and hand_result.hand_landmarks
                else []
            )

            detected_labels = []

            if hand_result is not None and hand_result.handedness:
                for hand_info in hand_result.handedness:
                    if hand_info:
                        detected_labels.append(
                            hand_info[0].category_name
                        )
                    else:
                        detected_labels.append("Unknown")

            while len(detected_labels) < len(detected_hands):
                detected_labels.append("Unknown")

            if len(detected_hands) > 0:
                hand_detected_frames += 1

            if len(detected_hands) >= 2:
                two_hand_detected_frames += 1

            normalized_hands = []

            for hand_lms in detected_hands:
                hand_array = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in hand_lms
                ], dtype=np.float32)

                hand_pts = (
                    hand_array - mid_shoulder
                ) / shoulder_width

                normalized_hands.append(hand_pts)

            pose_available = (
                pose_result is not None
                and bool(pose_result.pose_landmarks)
            )

            # Use body wrist landmarks for anatomical assignment.
            # X/Y are used to reduce differences in depth conventions.
            if pose_available:
                pose_lms = pose_result.pose_landmarks[0]

                left_wrist = np.array([
                    pose_lms[15].x,
                    pose_lms[15].y
                ], dtype=np.float32)

                right_wrist = np.array([
                    pose_lms[16].x,
                    pose_lms[16].y
                ], dtype=np.float32)

            else:
                left_wrist = None
                right_wrist = None

            # =================================================
            # TWO-HANDED SIGNS
            # =================================================
            if video_has_two_hands:

                if len(normalized_hands) >= 2:

                    hand_wrist_0 = np.array([
                        detected_hands[0][0].x,
                        detected_hands[0][0].y
                    ], dtype=np.float32)

                    hand_wrist_1 = np.array([
                        detected_hands[1][0].x,
                        detected_hands[1][0].y
                    ], dtype=np.float32)

                    if pose_available:
                        direct_distance = (
                            np.linalg.norm(
                                hand_wrist_0 - left_wrist
                            )
                            + np.linalg.norm(
                                hand_wrist_1 - right_wrist
                            )
                        )

                        crossed_distance = (
                            np.linalg.norm(
                                hand_wrist_0 - right_wrist
                            )
                            + np.linalg.norm(
                                hand_wrist_1 - left_wrist
                            )
                        )

                        if direct_distance <= crossed_distance:
                            frame_joints[33:54] = normalized_hands[0]
                            frame_joints[54:75] = normalized_hands[1]
                        else:
                            frame_joints[33:54] = normalized_hands[1]
                            frame_joints[54:75] = normalized_hands[0]

                    else:
                        # Fallback: MediaPipe handedness
                        assigned = set()

                        for i, hand_pts in enumerate(normalized_hands):
                            label = detected_labels[i]

                            if label == "Left" and "Left" not in assigned:
                                frame_joints[33:54] = hand_pts
                                assigned.add("Left")

                            elif label == "Right" and "Right" not in assigned:
                                frame_joints[54:75] = hand_pts
                                assigned.add("Right")

                            else:
                                # Deterministic fallback for unknown labels
                                if "Left" not in assigned:
                                    frame_joints[33:54] = hand_pts
                                    assigned.add("Left")
                                elif "Right" not in assigned:
                                    frame_joints[54:75] = hand_pts
                                    assigned.add("Right")

                elif len(normalized_hands) == 1:
                    # Keep the detected hand in the slot nearest
                    # its corresponding body wrist when possible.
                    hand_pts = normalized_hands[0]

                    if pose_available:
                        detected_wrist = np.array([
                            detected_hands[0][0].x,
                            detected_hands[0][0].y
                        ], dtype=np.float32)

                        left_distance = np.linalg.norm(
                            detected_wrist - left_wrist
                        )

                        right_distance = np.linalg.norm(
                            detected_wrist - right_wrist
                        )

                        if left_distance <= right_distance:
                            frame_joints[33:54] = hand_pts
                        else:
                            frame_joints[54:75] = hand_pts

                    elif detected_labels[0] == "Left":
                        frame_joints[33:54] = hand_pts
                    else:
                        frame_joints[54:75] = hand_pts

            # =================================================
            # SINGLE-HANDED SIGNS
            # Temporarily map detected hands to their likely
            # anatomical sides; choose the active side later.
            # =================================================
            else:
                if len(normalized_hands) >= 2 and pose_available:
                    # Assign both hands by body-wrist proximity.
                    hand_wrist_0 = np.array([
                        detected_hands[0][0].x,
                        detected_hands[0][0].y
                    ], dtype=np.float32)

                    hand_wrist_1 = np.array([
                        detected_hands[1][0].x,
                        detected_hands[1][0].y
                    ], dtype=np.float32)

                    direct_distance = (
                        np.linalg.norm(hand_wrist_0 - left_wrist)
                        + np.linalg.norm(hand_wrist_1 - right_wrist)
                    )

                    crossed_distance = (
                        np.linalg.norm(hand_wrist_0 - right_wrist)
                        + np.linalg.norm(hand_wrist_1 - left_wrist)
                    )

                    if direct_distance <= crossed_distance:
                        frame_joints[33:54] = normalized_hands[0]
                        frame_joints[54:75] = normalized_hands[1]
                    else:
                        frame_joints[33:54] = normalized_hands[1]
                        frame_joints[54:75] = normalized_hands[0]

                elif len(normalized_hands) >= 1:
                    # When only one hand is detected, use pose wrist
                    # proximity, falling back to handedness.
                    hand_pts = normalized_hands[0]

                    if pose_available:
                        detected_wrist = np.array([
                            detected_hands[0][0].x,
                            detected_hands[0][0].y
                        ], dtype=np.float32)

                        left_distance = np.linalg.norm(
                            detected_wrist - left_wrist
                        )

                        right_distance = np.linalg.norm(
                            detected_wrist - right_wrist
                        )

                        if left_distance <= right_distance:
                            frame_joints[33:54] = hand_pts
                        else:
                            frame_joints[54:75] = hand_pts

                    elif detected_labels[0] == "Left":
                        frame_joints[33:54] = hand_pts
                    else:
                        frame_joints[54:75] = hand_pts

            frame_joints_list.append(frame_joints)

        # ====================================================
        # SINGLE-HANDED: CHOOSE THE MORE ACTIVE HAND
        # Then place it in the Right slot for every frame.
        # ====================================================
        selected_hand = "Right"

        if not video_has_two_hands and frame_joints_list:
            left_motion = 0.0
            right_motion = 0.0
            left_pairs = 0
            right_pairs = 0

            for i in range(1, len(frame_joints_list)):
                previous_frame = frame_joints_list[i - 1]
                current_frame = frame_joints_list[i]

                previous_left = previous_frame[33:54]
                current_left = current_frame[33:54]

                previous_right = previous_frame[54:75]
                current_right = current_frame[54:75]

                # Compare motion only when both consecutive frames
                # contain a detected hand in that slot.
                if (
                    np.any(previous_left != 0)
                    and np.any(current_left != 0)
                ):
                    left_motion += np.mean(
                        np.linalg.norm(
                            current_left - previous_left,
                            axis=1
                        )
                    )
                    left_pairs += 1

                if (
                    np.any(previous_right != 0)
                    and np.any(current_right != 0)
                ):
                    right_motion += np.mean(
                        np.linalg.norm(
                            current_right - previous_right,
                            axis=1
                        )
                    )
                    right_pairs += 1

            left_score = (
                left_motion / left_pairs
                if left_pairs > 0 else 0.0
            )

            right_score = (
                right_motion / right_pairs
                if right_pairs > 0 else 0.0
            )

            # Choose the more active side only if it has usable
            # consecutive-frame observations.
            if left_pairs > 0 and (
                right_pairs == 0 or left_score > right_score
            ):
                selected_hand = "Left"

            for joints in frame_joints_list:
                if selected_hand == "Left":
                    joints[54:75] = joints[33:54].copy()

                # Single-handed signs use only the Right slot.
                joints[33:54] = 0.0

            print(f"Selected signing hand: {selected_hand}")
            print(f"Left movement score: {left_score:.6f}")
            print(f"Right movement score: {right_score:.6f}")

        # ====================================================
        # RESAMPLE TO EXACTLY 60 FRAMES
        # ====================================================
        total_len = len(frame_joints_list)

        if total_len == 0:
            print(f"Skipping '{video_file}': No processed frames.")
            continue

        indices = np.linspace(
            0,
            total_len - 1,
            SEQUENCE_LENGTH
        ).astype(int)

        resampled_sequence = np.array(
            [frame_joints_list[i] for i in indices],
            dtype=np.float32
        )

        # Validate output
        if resampled_sequence.shape != (
            SEQUENCE_LENGTH,
            TOTAL_JOINTS,
            3
        ):
            print(
                f"ERROR: Invalid output shape: "
                f"{resampled_sequence.shape}"
            )
            continue

        if not np.all(np.isfinite(resampled_sequence)):
            print(f"ERROR: NaN/Inf detected in {video_file}.")
            continue

        # Save one NPY per video
        np.save(output_path, resampled_sequence)

        total_frames = len(frames)
        hand_rate = hand_detected_frames / total_frames * 100
        pose_rate = pose_detected_frames / total_frames * 100
        two_hand_rate = two_hand_detected_frames / total_frames * 100

        mode = (
            "Two-Handed"
            if video_has_two_hands
            else "Single-Handed Force Right"
        )

        print(f"Saved: {output_path}")
        print(f"Mode: {mode}")
        print(f"Output shape: {resampled_sequence.shape}")
        print(f"Pose detection: {pose_rate:.1f}%")
        print(f"Hand detection: {hand_rate:.1f}%")
        print(f"Frames with 2 hands: {two_hand_rate:.1f}%")

        if hand_rate < 50:
            print(
                "WARNING: Low hand detection. "
                "Check lighting, camera angle, and hand visibility."
            )

        if pose_rate < 50:
            print(
                "WARNING: Low pose detection. "
                "Check upper-body visibility."
            )

        if video_has_two_hands and two_hand_rate < 30:
            print(
                "WARNING: Two-handed class but two hands were "
                "detected infrequently. Review this video."
            )

print("\nVideo dataset conversion complete!")
