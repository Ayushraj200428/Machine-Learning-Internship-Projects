from __future__ import annotations

import os
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TRAIN_DIR  = BASE_DIR / "train" / "train"
TEST_DIR   = BASE_DIR / "test1"
OUT_DIR    = BASE_DIR         

IMG_SIZE   = (64, 64)          
MAX_TRAIN  = 4000              
MAX_TEST   = 2000              
PCA_COMPONENTS = 150          
VAL_SIZE   = 0.20             
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray | None:
    """Load an image, convert to RGB, resize, return numpy array."""
    try:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def extract_hog(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to HOG feature vector."""
    gray = rgb2gray(img)         
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


def load_train_data(train_dir: Path, max_per_class: int):
  
    print("Loading training images …")
    cat_files = sorted(train_dir.glob("cat.*.jpg"))[:max_per_class]
    dog_files = sorted(train_dir.glob("dog.*.jpg"))[:max_per_class]

    print(f"  cats: {len(cat_files)}  |  dogs: {len(dog_files)}")

    X, y = [], []
    for label, files in [(0, cat_files), (1, dog_files)]:
        for fp in files:
            img = load_image(fp)
            if img is not None:
                X.append(extract_hog(img))
                y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_test_data(test_dir: Path, max_images: int):
    print("Loading test images …")
    files = sorted(test_dir.glob("*.jpg"),
                   key=lambda p: int(p.stem))[:max_images]
    ids, X = [], []
    for fp in files:
        img = load_image(fp)
        if img is not None:
            X.append(extract_hog(img))
            ids.append(int(fp.stem))

    return np.array(X, dtype=np.float32), ids


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── 1. Load & split data ──────────────────
    X, y = load_train_data(TRAIN_DIR, MAX_TRAIN)
    print(f"Feature matrix shape: {X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    # ── 2. Pipeline: Scaler → PCA → SVM ───────
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)),
        ("svm",    SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ])

    # ── 3. Hyperparameter search ───────────────
    param_grid = {
        "svm__C":     [0.1, 1, 10],
        "svm__gamma": ["scale", 0.001, 0.01],
    }
    print("\nRunning GridSearchCV (this may take several minutes) …")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    print(f"\nBest params : {gs.best_params_}")
    print(f"Best CV acc : {gs.best_score_:.4f}")

    # ── 4. Validate ────────────────────────────
    y_pred      = best_model.predict(X_val)
    y_prob      = best_model.predict_proba(X_val)[:, 1]
    val_acc     = accuracy_score(y_val, y_pred)
    val_auc     = roc_auc_score(y_val, y_prob)

    print(f"\nValidation accuracy : {val_acc:.4f}")
    print(f"Validation AUC-ROC  : {val_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Cat", "Dog"]))

    # ── 5. Plots ───────────────────────────────
    save_results_plot(gs, y_val, y_pred, y_prob, val_acc, val_auc, OUT_DIR)

    # ── 6. Predict test set → submission CSV ──
    X_test, test_ids = load_test_data(TEST_DIR, MAX_TEST)
    if len(X_test):
        test_preds = best_model.predict(X_test)
        test_labels = ["dog" if p == 1 else "cat" for p in test_preds]

        submission = pd.DataFrame({
            "id":    test_ids,
            "label": test_labels,
        })
        submission_path = OUT_DIR / "submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"\nSubmission saved → {submission_path}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed/60:.1f} min")


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def save_results_plot(gs, y_val, y_pred, y_prob, val_acc, val_auc, out_dir: Path):
   
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("SVM Cats vs Dogs – Results Dashboard", fontsize=16, fontweight="bold", y=0.98)
    gs_layout = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── (a) Confusion Matrix ──────────────────
    ax1 = fig.add_subplot(gs_layout[0, 0])
    cm  = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    disp.plot(ax=ax1, colorbar=False, cmap="Blues")
    ax1.set_title("Confusion Matrix", fontweight="bold")

    # ── (b) ROC Curve ─────────────────────────
    ax2 = fig.add_subplot(gs_layout[0, 1])
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    ax2.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"AUC = {val_auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve", fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    # ── (c) GridSearch heatmap (C × gamma) ────
    ax3 = fig.add_subplot(gs_layout[0, 2])
    results = pd.DataFrame(gs.cv_results_)

    pivot_rows = results[["param_svm__C", "param_svm__gamma", "mean_test_score"]].copy()
    pivot_rows["param_svm__C"]     = pivot_rows["param_svm__C"].astype(str)
    pivot_rows["param_svm__gamma"] = pivot_rows["param_svm__gamma"].astype(str)
    pivot = pivot_rows.pivot(index="param_svm__gamma",
                             columns="param_svm__C",
                             values="mean_test_score")
    im = ax3.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels([f"C={c}" for c in pivot.columns], fontsize=8)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels([f"γ={g}" for g in pivot.index], fontsize=8)
    ax3.set_title("GridSearch CV Accuracy", fontweight="bold")
    fig.colorbar(im, ax=ax3, shrink=0.8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax3.text(j, i, f"{pivot.values[i,j]:.3f}",
                     ha="center", va="center", fontsize=7, color="black")

    # ── (d) Per-class precision / recall / F1 ─
    ax4 = fig.add_subplot(gs_layout[1, 0])
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics = {
        "Precision": [
            precision_score(y_val, y_pred, pos_label=0),
            precision_score(y_val, y_pred, pos_label=1),
        ],
        "Recall": [
            recall_score(y_val, y_pred, pos_label=0),
            recall_score(y_val, y_pred, pos_label=1),
        ],
        "F1-Score": [
            f1_score(y_val, y_pred, pos_label=0),
            f1_score(y_val, y_pred, pos_label=1),
        ],
    }
    x  = np.arange(2)
    w  = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (metric, vals) in enumerate(metrics.items()):
        ax4.bar(x + i * w, vals, w, label=metric, color=colors[i])
    ax4.set_xticks(x + w)
    ax4.set_xticklabels(["Cat", "Dog"])
    ax4.set_ylim(0, 1.1)
    ax4.set_ylabel("Score")
    ax4.set_title("Per-class Metrics", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)

    # ── (e) Prediction probability distribution
    ax5 = fig.add_subplot(gs_layout[1, 1])
    ax5.hist(y_prob[y_val == 0], bins=30, alpha=0.6, color="steelblue", label="Cat (true)")
    ax5.hist(y_prob[y_val == 1], bins=30, alpha=0.6, color="salmon",    label="Dog (true)")
    ax5.axvline(0.5, color="k", linestyle="--", lw=1)
    ax5.set_xlabel("Predicted probability (dog)")
    ax5.set_ylabel("Count")
    ax5.set_title("Probability Distribution", fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # ── (f) Summary text box ──────────────────
    ax6 = fig.add_subplot(gs_layout[1, 2])
    ax6.axis("off")
    summary = (
        f"Model Summary\n"
        f"{'─'*28}\n"
        f"Algorithm   : SVM (RBF kernel)\n"
        f"Features    : HOG + PCA-{PCA_COMPONENTS}\n"
        f"Image size  : {IMG_SIZE[0]}×{IMG_SIZE[1]} px\n"
        f"Train size  : {MAX_TRAIN*2} images\n"
        f"Val size    : {int(MAX_TRAIN*2*VAL_SIZE)} images\n"
        f"{'─'*28}\n"
        f"Best C      : {gs.best_params_['svm__C']}\n"
        f"Best gamma  : {gs.best_params_['svm__gamma']}\n"
        f"CV accuracy : {gs.best_score_:.4f}\n"
        f"Val accuracy: {val_acc:.4f}\n"
        f"Val AUC-ROC : {val_auc:.4f}\n"
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=10, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    out_path = out_dir / "model_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Results plot saved → {out_path}")


if __name__ == "__main__":
    main()
