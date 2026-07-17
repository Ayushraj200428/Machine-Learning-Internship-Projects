# ── Imports ──────────────────────────────────────────────────────────────────
import os, glob, random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "leapGestRecog")

IMG_H, IMG_W = 64, 64
BATCH_SIZE   = 64
EPOCHS       = 20
SEED         = 42

GESTURE_FOLDERS = [
    "01_palm", "02_l", "03_fist", "04_fist_moved",
    "05_thumb", "06_index", "07_ok", "08_palm_moved",
    "09_c", "10_down"
]
CLASS_NAMES = [
    "Palm", "L", "Fist", "Fist Moved",
    "Thumb", "Index", "OK", "Palm Moved",
    "C", "Down"
]
NUM_CLASSES = len(CLASS_NAMES)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Collect file paths (fast: no image reading yet)
# ─────────────────────────────────────────────────────────────────────────────
print("Scanning dataset folder for image paths …")
all_paths, all_labels = [], []

for subject in sorted(os.listdir(DATA_DIR)):
    subject_dir = os.path.join(DATA_DIR, subject)
    if not os.path.isdir(subject_dir):
        continue
    for label_idx, gesture in enumerate(GESTURE_FOLDERS):
        gesture_dir = os.path.join(subject_dir, gesture)
        if not os.path.isdir(gesture_dir):
            continue
        for img_path in glob.glob(os.path.join(gesture_dir, "*.png")):
            all_paths.append(img_path)
            all_labels.append(label_idx)

print(f"  Found {len(all_paths)} images across {NUM_CLASSES} classes")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Split paths into train / val / test
# ─────────────────────────────────────────────────────────────────────────────
paths  = np.array(all_paths)
labels = np.array(all_labels)

paths_train, paths_tmp, labels_train, labels_tmp = train_test_split(
    paths, labels, test_size=0.30, random_state=SEED, stratify=labels
)
paths_val, paths_test, labels_val, labels_test = train_test_split(
    paths_tmp, labels_tmp, test_size=0.50, random_state=SEED, stratify=labels_tmp
)

print(f"  Train: {len(paths_train)}  |  Val: {len(paths_val)}  |  Test: {len(paths_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — tf.data pipeline  (reads & preprocesses images on-the-fly)
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(path, label):
    """Read one PNG, decode as grayscale, resize, normalize → [0,1]."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_png(raw, channels=1)         
    image = tf.image.resize(image, [IMG_H, IMG_W])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(paths_train, labels_train, shuffle=True)
val_ds   = make_dataset(paths_val,   labels_val)
test_ds  = make_dataset(paths_test,  labels_test)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Visualise one sample per class (saved to file, no GUI needed)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSaving sample images …")

sample_images, sample_labels = next(iter(
    tf.data.Dataset.from_tensor_slices((paths_train, labels_train))
    .shuffle(5000, seed=SEED)
    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(500)
))
sample_images = sample_images.numpy()
sample_labels = np.argmax(sample_labels.numpy(), axis=1)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample Images — One Per Gesture Class", fontsize=14)
for cls in range(NUM_CLASSES):
    idxs = np.where(sample_labels == cls)[0]
    ax   = axes[cls // 5][cls % 5]
    if len(idxs):
        ax.imshow(sample_images[idxs[0]].squeeze(), cmap="gray")
    ax.set_title(CLASS_NAMES[cls], fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "sample_images.png"), dpi=100)
plt.close()
print("  Saved → sample_images.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build CNN model
# ─────────────────────────────────────────────────────────────────────────────
"""
Architecture:
  Input  64×64×1  (grayscale)
  Block1: Conv32  → BN → MaxPool → Dropout(0.25)
  Block2: Conv64  → BN → MaxPool → Dropout(0.25)
  Block3: Conv128 → BN → MaxPool → Dropout(0.25)
  Flatten → Dense(256) → Dropout(0.5) → Dense(10, softmax)
"""

def build_cnn():
    model = models.Sequential([
        
        layers.Conv2D(32, (3,3), activation="relu", padding="same",
                      input_shape=(IMG_H, IMG_W, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

  
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

      
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.50),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_cnn()
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Train
# ─────────────────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
]

print("\nTraining …")
t0 = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)
print(f"Training finished in {(time.time()-t0)/60:.1f} min")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Plot training history
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Training History", fontsize=14)

ax1.plot(history.history["accuracy"],     label="Train")
ax1.plot(history.history["val_accuracy"], label="Val")
ax1.set_title("Accuracy");  ax1.set_xlabel("Epoch");  ax1.legend();  ax1.grid(True)

ax2.plot(history.history["loss"],     label="Train")
ax2.plot(history.history["val_loss"], label="Val")
ax2.set_title("Loss");  ax2.set_xlabel("Epoch");  ax2.legend();  ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "training_history.png"), dpi=100)
plt.close()
print("Saved → training_history.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Evaluate on test set
# ─────────────────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

y_true_list, y_pred_list = [], []
for imgs, lbls in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_pred_list.extend(np.argmax(preds, axis=1))
    y_true_list.extend(np.argmax(lbls.numpy(), axis=1))

y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted");  plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=100)
plt.close()
print("Saved → confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Save model
# ─────────────────────────────────────────────────────────────────────────────
save_path = os.path.join(BASE_DIR, "gesture_model.keras")
model.save(save_path)
print(f"Model saved → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Predict on a single image (demo)
# ─────────────────────────────────────────────────────────────────────────────
def predict_gesture(image_path):
    """Load one image and print the predicted gesture + confidence bar."""
    raw   = tf.io.read_file(image_path)
    image = tf.image.decode_png(raw, channels=1)
    image = tf.image.resize(image, [IMG_H, IMG_W])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, 0)           

    probs      = model.predict(image, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    confidence = probs[pred_idx] * 100

    img_np = image.numpy()[0].squeeze()
    plt.figure(figsize=(4, 4))
    plt.imshow(img_np, cmap="gray")
    plt.title(f"Predicted: {CLASS_NAMES[pred_idx]}  ({confidence:.1f}%)", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "single_prediction.png"), dpi=100)
    plt.close()

    print(f"\nPredicted Gesture : {CLASS_NAMES[pred_idx]}")
    print(f"Confidence        : {confidence:.1f}%")
    print("\nAll class probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        bar = "█" * int(p * 30)
        print(f"  {name:<14} {p*100:5.1f}%  {bar}")
    print("Saved → single_prediction.png")

# Use a random test image as the demo
demo_path = str(paths_test[random.randint(0, len(paths_test)-1)])
print(f"\n── Single Image Prediction Demo ──\nImage: {demo_path}")
predict_gesture(demo_path)
