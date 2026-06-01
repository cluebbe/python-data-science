"""
Workshop Tutorial: K-Means Clustering with scikit-learn
=======================================================

K-means is an unsupervised learning algorithm that partitions data
into K clusters. Unlike supervised learning, there are no labels —
the algorithm discovers structure in the data on its own.

Each cluster is defined by its centroid (the mean of all points in it).
The algorithm alternates between two steps until convergence:
  1. Assign each point to its nearest centroid
  2. Recompute each centroid as the mean of all assigned points

We'll use make_blobs to generate a clean synthetic dataset with known
cluster structure, so we can verify the algorithm finds what we put in.
"""

# ------------------------------------------------------------
# Step 0 — Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------
# Step 1 — Generate & explore the data
# ------------------------------------------------------------
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
# X: shape (300, 2) — two features, ready to plot directly
# y: true cluster labels (0, 1, 2) — hidden during training, used for evaluation

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"True cluster sizes: {np.bincount(y)}\n")

# Visualise the raw data without labels
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolors="k", linewidths=0.3)
ax.set_title("Raw Data (no labels shown)")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 2 — Scale the features
# ------------------------------------------------------------
# K-means uses Euclidean distance — features with larger numeric ranges
# dominate the distance calculation. Scaling ensures all features
# contribute equally.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== Feature Scaling ===")
print(f"Before — mean: {X[:, 0].mean():.3f}  std: {X[:, 0].std():.3f}")
print(f"After  — mean: {X_scaled[:, 0].mean():.3f}  std: {X_scaled[:, 0].std():.3f}\n")

# ------------------------------------------------------------
# Step 3 — Choose K: elbow + silhouette
# ------------------------------------------------------------
# Elbow: inertia always decreases with K — look for where the rate
# of decrease sharply slows (the "elbow").
# Silhouette: measures cluster quality directly. Range -1 to 1.
# Higher is better. Pick the K with the highest score.
# Both methods should agree — silhouette gives an objective maximum,
# elbow gives visual intuition.

print("=== Choosing K ===")
k_range = range(2, 11)
inertias, sil_scores = [], []

for k in k_range:
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km_k.fit_predict(X_scaled)
    inertias.append(km_k.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels_k))
    print(f"  k={k:2d}  inertia={km_k.inertia_:7.2f}  silhouette={sil_scores[-1]:.4f}")

best_k = k_range.start + int(np.argmax(sil_scores))
print(f"\n  Best K by silhouette: {best_k}\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(k_range, inertias, marker="o")
axes[0].set_xlabel("Number of clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(k_range, sil_scores, marker="o")
axes[1].set_xlabel("Number of clusters (K)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette Method")
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Choosing K")
fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 4 — Fit K-means with K=3
# ------------------------------------------------------------
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_scaled)
labels = km.labels_

print("=== K-Means (k=3) ===")
print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Inertia:       {km.inertia_:.4f}\n")

# ------------------------------------------------------------
# Step 5 — Visualise clusters
# ------------------------------------------------------------
# make_blobs generates 2-D data — no PCA needed, plot directly.

colors = ["steelblue", "tomato", "seagreen"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 5a: Clusters found by K-means
for cluster in range(3):
    mask = labels == cluster
    axes[0].scatter(
        X_scaled[mask, 0], X_scaled[mask, 1],
        c=colors[cluster], label=f"Cluster {cluster}",
        alpha=0.7, edgecolors="k", linewidths=0.3,
    )
axes[0].scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    c="black", marker="X", s=120, label="Centroid", zorder=5,
)
axes[0].set_title("K-Means Clusters")
axes[0].set_xlabel("Feature 1 (scaled)")
axes[0].set_ylabel("Feature 2 (scaled)")
axes[0].legend()

# 5b: True labels for comparison
for cluster in range(3):
    mask = y == cluster
    axes[1].scatter(
        X_scaled[mask, 0], X_scaled[mask, 1],
        c=colors[cluster], label=f"Origin {cluster}",
        alpha=0.7, edgecolors="k", linewidths=0.3,
    )
axes[1].set_title("Ground Truth (known only because we generated the data)")
axes[1].set_xlabel("Feature 1 (scaled)")
axes[1].set_ylabel("Feature 2 (scaled)")
axes[1].legend()

fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 6 — Effect of K
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k in zip(axes, [2, 3, 5]):
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km_k.fit_predict(X_scaled)
    sil_k = silhouette_score(X_scaled, labels_k)
    for cluster in range(k):
        mask = labels_k == cluster
        ax.scatter(
            X_scaled[mask, 0], X_scaled[mask, 1],
            alpha=0.7, edgecolors="k", linewidths=0.3,
        )
    ax.scatter(
        km_k.cluster_centers_[:, 0], km_k.cluster_centers_[:, 1],
        c="black", marker="X", s=120, zorder=5,
    )
    ax.set_title(f"K={k}  |  silhouette={sil_k:.3f}")
    ax.set_xlabel("Feature 1 (scaled)")
    ax.set_ylabel("Feature 2 (scaled)")

fig.suptitle("Effect of K on Clustering")
fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 7 — Make a single prediction (assign new sample)
# ------------------------------------------------------------
sample = np.array([[0.0, 4.0]])          # a new, unseen point
sample_scaled = scaler.transform(sample)
cluster_id = km.predict(sample_scaled)[0]
distances = km.transform(sample_scaled)[0]

print("=== Single Prediction ===")
print(f"Input: Feature1=0.0, Feature2=4.0")
print(f"Assigned to cluster: {cluster_id}")
print("Distances to all centroids:")
for i, d in enumerate(distances):
    marker = " ← assigned" if i == cluster_id else ""
    print(f"  Centroid {i}: {d:.4f}{marker}")
