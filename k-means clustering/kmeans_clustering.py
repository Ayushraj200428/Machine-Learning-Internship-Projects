import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────
# Helper: safe integer prompt
# ─────────────────────────────────────────────
def ask_int(prompt, min_val=1, max_val=None):
    while True:
        try:
            val = int(input(prompt).strip())
            if val < min_val:
                print(f"  Please enter a value >= {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Please enter a value <= {max_val}.")
                continue
            return val
        except ValueError:
            print("  Invalid input. Please enter a whole number.")


# ═════════════════════════════════════════════
# STEP 1 – Load Dataset
# ═════════════════════════════════════════════
print("=" * 55)
print("   K-MEANS CUSTOMER SEGMENTATION")
print("=" * 55)

while True:
    csv_path = input("\nEnter path to CSV file (e.g. Mall_Customers.csv): ").strip()
    if os.path.isfile(csv_path):
        break
    print(f"  File not found: '{csv_path}'. Please try again.")

df = pd.read_csv(csv_path)

print(f"\nDataset loaded successfully!")
print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumns available:")
for i, col in enumerate(df.columns):
    print(f"  [{i}] {col}  (dtype: {df[col].dtype})")

print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nBasic statistics:")
print(df.describe().to_string())


# ═════════════════════════════════════════════
# STEP 2 – Choose Features
# ═════════════════════════════════════════════
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumeric columns available for clustering:")
for i, col in enumerate(numeric_cols):
    print(f"  [{i}] {col}")

print("\nSelect feature columns for clustering.")
print("Enter column numbers separated by commas (e.g. 2,3 or 1,2,3).")
print("At least 2 features are required.")

while True:
    raw = input("Your selection: ").strip()
    try:
        indices = [int(x.strip()) for x in raw.split(",")]
        if len(indices) < 2:
            print("  Please select at least 2 features.")
            continue
        if any(i < 0 or i >= len(numeric_cols) for i in indices):
            print(f"  Index out of range. Choose between 0 and {len(numeric_cols)-1}.")
            continue
        selected_features = [numeric_cols[i] for i in indices]
        break
    except ValueError:
        print("  Invalid input. Enter comma-separated numbers.")

print(f"\nSelected features: {selected_features}")

X = df[selected_features].values

# ═════════════════════════════════════════════
# STEP 3 – Scale Features
# ═════════════════════════════════════════════
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeatures standardised (zero mean, unit variance).")


# ═════════════════════════════════════════════
# STEP 4 – Auto-detect or manually choose k
# ═════════════════════════════════════════════
print("\n" + "-" * 55)
print("How would you like to choose the number of clusters (k)?")
print("  [1] Let the program find the optimal k automatically")
print("  [2] I will specify k manually")
mode = ask_int("Enter 1 or 2: ", min_val=1, max_val=2)

max_k = min(10, df.shape[0] - 1)  

if mode == 1:
    # ── Auto: Elbow + Silhouette ──────────────
    print(f"\nRunning Elbow & Silhouette analysis for k = 2 to {max_k} ...")
    inertias, silhouettes = [], []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    # Plot elbow + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optimal k Selection", fontsize=14, fontweight="bold")

    axes[0].plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, silhouettes, "rs-", linewidth=2, markersize=8)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.show()

    best_k = k_range[int(np.argmax(silhouettes))]
    print(f"\nSilhouette scores: { {k: round(s,4) for k,s in zip(k_range, silhouettes)} }")
    print(f"Best k by silhouette score: {best_k}")

    use_best = input(f"\nUse k={best_k}? Press Enter to confirm, or type a different k: ").strip()
    if use_best == "":
        OPTIMAL_K = best_k
    else:
        OPTIMAL_K = ask_int("Enter your preferred k: ", min_val=2, max_val=max_k)
else:
    # ── Manual ───────────────────────────────
    OPTIMAL_K = ask_int(f"\nEnter number of clusters k (2–{max_k}): ", min_val=2, max_val=max_k)

print(f"\nUsing k = {OPTIMAL_K}")


# ═════════════════════════════════════════════
# STEP 5 – Fit Final K-Means Model
# ═════════════════════════════════════════════
kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df["Cluster"])
print(f"\nModel trained.")
print(f"  Silhouette Score : {sil:.4f}")
print(f"  Inertia (WCSS)   : {kmeans.inertia_:.2f}")


# ═════════════════════════════════════════════
# STEP 6 – Cluster Profiles & Labelling
# ═════════════════════════════════════════════
cluster_summary = (
    df.groupby("Cluster")[selected_features]
    .mean()
    .round(2)
)
cluster_summary["Count"] = df.groupby("Cluster").size()

f1, f2 = selected_features[0], selected_features[1]
med_f1 = df[f1].median()
med_f2 = df[f2].median()

segment_map = {}
for c, row in cluster_summary.iterrows():
    high1 = row[f1] >= med_f1
    high2 = row[f2] >= med_f2
    if high1 and high2:
        segment_map[c] = f"High {f1} – High {f2}"
    elif high1 and not high2:
        segment_map[c] = f"High {f1} – Low {f2}"
    elif not high1 and high2:
        segment_map[c] = f"Low {f1} – High {f2}"
    else:
        segment_map[c] = f"Low {f1} – Low {f2}"

cluster_summary["Segment"] = cluster_summary.index.map(segment_map)

print("\nCluster Summary:")
print(cluster_summary.to_string())


# ═════════════════════════════════════════════
# STEP 7 – Visualisations
# ═════════════════════════════════════════════
COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#2980b9", "#8e44ad", "#27ae60",
]
colors = COLORS[:OPTIMAL_K]

# ── 7a. 2D scatter: first two selected features ──
fig, ax = plt.subplots(figsize=(10, 7))
for c in range(OPTIMAL_K):
    mask = df["Cluster"] == c
    ax.scatter(
        df.loc[mask, f1], df.loc[mask, f2],
        c=colors[c], label=f"Cluster {c}: {segment_map[c]}",
        s=80, edgecolors="white", linewidth=0.5, alpha=0.85,
    )

centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(
    centroids_orig[:, 0], centroids_orig[:, 1],
    c="black", s=200, marker="X", zorder=5, label="Centroids",
)
ax.set_title(f"Customer Segments – {f1} vs {f2}", fontsize=14, fontweight="bold")
ax.set_xlabel(f1, fontsize=12)
ax.set_ylabel(f2, fontsize=12)
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clusters_2d.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 7b. 3D scatter (only if user picked 3+ features) ──
if len(selected_features) >= 3:
    f3 = selected_features[2]
    fig3d = plt.figure(figsize=(11, 8))
    ax3 = fig3d.add_subplot(111, projection="3d")
    for c in range(OPTIMAL_K):
        mask = df["Cluster"] == c
        ax3.scatter(
            df.loc[mask, f1], df.loc[mask, f2], df.loc[mask, f3],
            c=colors[c], s=60, alpha=0.8, label=f"Cluster {c}",
        )
    ax3.set_xlabel(f1)
    ax3.set_ylabel(f2)
    ax3.set_zlabel(f3)
    ax3.set_title(f"3D Segments: {f1}, {f2}, {f3}", fontsize=12, fontweight="bold")
    ax3.legend()
    plt.tight_layout()
    plt.savefig("clusters_3d.png", dpi=150, bbox_inches="tight")
    plt.show()

# ── 7c. Cluster size bar chart ──
fig, ax = plt.subplots(figsize=(max(8, OPTIMAL_K * 2), 5))
counts = df["Cluster"].value_counts().sort_index()
bars = ax.bar(
    [f"Cluster {i}\n{segment_map[i]}" for i in counts.index],
    counts.values,
    color=colors, edgecolor="black", linewidth=0.7,
)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontweight="bold")
ax.set_title("Number of Customers per Cluster", fontsize=14, fontweight="bold")
ax.set_ylabel("Customer Count")
ax.set_ylim(0, counts.max() + 10)
ax.grid(axis="y", alpha=0.3)
plt.xticks(fontsize=8)
plt.tight_layout()
plt.savefig("cluster_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 7d. Feature mean per cluster (grouped bar) ──
means = df.groupby("Cluster")[selected_features].mean()
x = np.arange(OPTIMAL_K)
width = 0.8 / len(selected_features)

fig, ax = plt.subplots(figsize=(12, 6))
for i, feat in enumerate(selected_features):
    ax.bar(x + i * width, means[feat], width, label=feat, alpha=0.85)
ax.set_title("Average Feature Values per Cluster", fontsize=14, fontweight="bold")
ax.set_xlabel("Cluster")
ax.set_ylabel("Mean Value")
ax.set_xticks(x + width * (len(selected_features) - 1) / 2)
ax.set_xticklabels([f"Cluster {i}" for i in range(OPTIMAL_K)])
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("cluster_features.png", dpi=150, bbox_inches="tight")
plt.show()


# ═════════════════════════════════════════════
# STEP 8 – Save Results
# ═════════════════════════════════════════════
output_name = input("\nEnter output CSV filename (default: clustered_customers.csv): ").strip()
if output_name == "":
    output_name = "clustered_customers.csv"
if not output_name.endswith(".csv"):
    output_name += ".csv"

df["Segment"] = df["Cluster"].map(segment_map)
df.to_csv(output_name, index=False)

print(f"\nResults saved to: {output_name}")
print("Plots saved : clusters_2d.png, cluster_distribution.png, cluster_features.png")
if len(selected_features) >= 3:
    print("             clusters_3d.png")
if mode == 1:
    print("             elbow_silhouette.png")
print("\nDone!")
