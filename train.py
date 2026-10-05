import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_DIR = "datasets"

SEQUENCE_LENGTH = 60
TOTAL_JOINTS = 75
COORDINATES = 3

INPUT_SHAPE = (
    SEQUENCE_LENGTH,
    TOTAL_JOINTS,
    COORDINATES
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

X = []
y = []

classes = sorted([
    d
    for d in os.listdir(DATASET_DIR)
    if os.path.isdir(
        os.path.join(DATASET_DIR, d)
    )
])


for idx, class_name in enumerate(classes):

    class_path = os.path.join(
        DATASET_DIR,
        class_name
    )


    for file_name in sorted(
        os.listdir(class_path)
    ):

        if not file_name.endswith(".npy"):
            continue


        file_path = os.path.join(
            class_path,
            file_name
        )


        try:

            data = np.load(file_path)

        except Exception as e:

            print(
                f"Skipping unreadable file: "
                f"{file_path}"
            )

            print(e)

            continue


        # ----------------------------------------------------
        # Strict shape check
        # ----------------------------------------------------

        if data.shape != INPUT_SHAPE:

            print(
                f"Skipping wrong shape: "
                f"{file_name} -> {data.shape}"
            )

            continue


        # ----------------------------------------------------
        # NaN / Inf check
        # ----------------------------------------------------

        if not np.isfinite(data).all():

            print(
                f"Skipping NaN/Inf file: "
                f"{file_name}"
            )

            continue


        X.append(
            data.astype(np.float32)
        )

        y.append(idx)


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)


np.save(
    "classes.npy",
    np.array(classes)
)


print(
    f"Loaded {len(X)} samples across "
    f"{len(classes)} classes:"
)

print(classes)


if len(X) == 0:

    raise ValueError(
        "No valid training samples found. "
        "Check datasets and .npy shapes."
    )


# ============================================================
# 3. SKELETON GRAPH & DEGREE NORMALIZATION (A_tilde = D^(-1/2) * A * D^(-1/2))
# ============================================================

def build_fsl_adjacency_matrix(
    num_joints=75
):

    A = np.eye(
        num_joints,
        dtype=np.float32
    )


    # POSE EDGES
    pose_edges = [
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24)
    ]

    for u, v in pose_edges:
        A[u, v] = 1.0
        A[v, u] = 1.0


    # HAND EDGES
    hand_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    # Left Hand
    for u, v in hand_edges:
        A[33 + u, 33 + v] = 1.0
        A[33 + v, 33 + u] = 1.0

    # Right Hand
    for u, v in hand_edges:
        A[54 + u, 54 + v] = 1.0
        A[54 + v, 54 + u] = 1.0


    # WRIST ↔ HAND CONNECTIONS
    A[15, 33] = 1.0
    A[33, 15] = 1.0

    A[16, 54] = 1.0
    A[54, 16] = 1.0


    # DEGREE NORMALIZATION FORMULA
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = deg[nonzero] ** -0.5
    D_mat = np.diag(deg_inv_sqrt)

    A_normalized = D_mat @ A @ D_mat

    return A_normalized.astype(np.float32)


# ============================================================
# 4. GRAPH CONVOLUTION (H^(l+1) = A_tilde * X * W)
# ============================================================

@tf.keras.utils.register_keras_serializable()
class TFLiteGraphConv(layers.Layer):

    def __init__(
        self,
        out_channels,
        adj_matrix,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.out_channels = out_channels

        self.adj_matrix = adj_matrix

        self.A = tf.constant(
            adj_matrix,
            dtype=tf.float32
        )


    def build(self, input_shape):

        self.dense = layers.Dense(
            self.out_channels,
            use_bias=False
        )

        super().build(input_shape)


    def get_config(self):

        config = super().get_config()

        config.update({
            "out_channels": self.out_channels,
            "adj_matrix": self.adj_matrix.tolist()
        })

        return config


    def call(self, inputs):

        # XW (Linear Projection)
        x_weight = self.dense(inputs)

        shape = tf.shape(inputs)
        batch_size = shape[0]
        frames = shape[1]

        x_reshaped = tf.reshape(
            x_weight,
            [
                -1,
                TOTAL_JOINTS,
                self.out_channels
            ]
        )

        # Graph Propagation: A_tilde * (XW)
        x_graph = tf.matmul(
            self.A,
            x_reshaped
        )

        return tf.reshape(
            x_graph,
            [
                batch_size,
                frames,
                TOTAL_JOINTS,
                self.out_channels
            ]
        )


# ============================================================
# 5. BUILD ST-GCN MODEL
# ============================================================

def build_stgcn_model(
    input_shape=INPUT_SHAPE,
    num_classes=None
):

    if num_classes is None:
        num_classes = len(classes)

    adj_matrix = build_fsl_adjacency_matrix(
        TOTAL_JOINTS
    )

    inputs = layers.Input(
        shape=input_shape
    )

    # BLOCK 1
    x = layers.BatchNormalization()(inputs)
    x = TFLiteGraphConv(32, adj_matrix)(x)
    x = layers.Conv2D(
        32,
        kernel_size=(9, 1),
        padding="same",
        activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # BLOCK 2
    x = TFLiteGraphConv(64, adj_matrix)(x)
    x = layers.Conv2D(
        64,
        kernel_size=(9, 1),
        padding="same",
        activation="relu"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # GLOBAL READOUT
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    return models.Model(
        inputs=inputs,
        outputs=outputs
    )


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

def augment_data(
    X_data,
    y_data
):

    X_aug = list(X_data)
    y_aug = list(y_data)

    for i in range(len(X_data)):
        sample = X_data[i].copy()

        # 1. SPATIAL SCALING
        scale_factor = np.random.uniform(
            0.85,
            1.15
        )
        augmented = sample * scale_factor

        # 2. TEMPORAL STRETCHING
        stretch_factor = np.random.uniform(
            0.85,
            1.15
        )
        original_frames = augmented.shape[0]
        target_frames = max(
            10,
            int(original_frames * stretch_factor)
        )

        old_positions = np.linspace(
            0,
            1,
            original_frames
        )
        new_positions = np.linspace(
            0,
            1,
            target_frames
        )

        stretched = np.zeros(
            (
                target_frames,
                TOTAL_JOINTS,
                COORDINATES
            ),
            dtype=np.float32
        )

        for joint in range(TOTAL_JOINTS):
            for coord in range(COORDINATES):
                stretched[:, joint, coord] = np.interp(
                    new_positions,
                    old_positions,
                    augmented[:, joint, coord]
                )

        final_positions = np.linspace(
            0,
            1,
            SEQUENCE_LENGTH
        )
        final_sample = np.zeros(
            INPUT_SHAPE,
            dtype=np.float32
        )

        for joint in range(TOTAL_JOINTS):
            for coord in range(COORDINATES):
                final_sample[:, joint, coord] = np.interp(
                    final_positions,
                    new_positions,
                    stretched[:, joint, coord]
                )

        # 3. GAUSSIAN NOISE
        valid_mask = (
            np.abs(final_sample).sum(
                axis=2,
                keepdims=True
            ) > 1e-8
        )
        noise = np.random.normal(
            0,
            0.003,
            final_sample.shape
        ).astype(np.float32)

        final_sample += noise * valid_mask

        X_aug.append(final_sample)
        y_aug.append(y_data[i])

    return (
        np.array(X_aug, dtype=np.float32),
        np.array(y_aug, dtype=np.int64)
    )


# ============================================================
# 7. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=(
        y
        if len(classes) > 1
        else None
    )
)


print(f"Training samples before augmentation: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# ============================================================
# 8. AUGMENT TRAINING DATA ONLY
# ============================================================

X_train, y_train = augment_data(
    X_train,
    y_train
)

print(f"Training samples after augmentation: {len(X_train)}")


# ============================================================
# 9. CLASS WEIGHTS
# ============================================================

unique_classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes,
    y=y_train
)

class_weights_dict = {
    int(class_id): float(weight)
    for class_id, weight
    in zip(
        unique_classes,
        class_weights
    )
}

print("Computed Class Weights:")
print(class_weights_dict)


# ============================================================
# 10. BUILD MODEL
# ============================================================

model = build_stgcn_model(
    num_classes=len(classes)
)

model.summary()


# ============================================================
# 11. COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 12. CALLBACKS
# ============================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# 13. TRAIN
# ============================================================

print("\nTraining ST-GCN Network...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=150,
    batch_size=8,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

model.save(
    "fsl_model.keras"
)

print("\nST-GCN training complete!")
print("Model saved to: fsl_model.keras")
