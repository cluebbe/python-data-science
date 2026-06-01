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

We'll use the Iris dataset — 150 flower samples, 4 features.
The true species labels are hidden during training and only used for evaluation.
"""

# ------------------------------------------------------------
# Step 0 — Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ------------------------------------------------------------
# Step 1 — Load & explore the data
# ------------------------------------------------------------
iris = load_iris()

X = iris.data    # shape: (150, 4) — we cluster on these
y = iris.target  # 0=setosa, 1=versicolor, 2=virginica — hidden during training

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}  ->  {list(iris.feature_names)}")
print(f"True classes (hidden during training): {iris.target_names}")
print(f"Class distribution: {np.bincount(y)}\n")

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
# Step 3 — Choose K: the Elbow method
# ------------------------------------------------------------
# Inertia = sum of squared distances from each point to its centroid.
# It always decreases as K grows. The "elbow" — where the rate of
# decrease sharply slows — suggests a good value of K.

print("=== Elbow Method ===")
k_range = range(1, 11)
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  k={k:2d}  inertia={km.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_range, inertias, marker="o")
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method — Choosing K")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 4 — Train K-means with K=3
# ------------------------------------------------------------
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_scaled)
labels = km.labels_   # cluster assignment for each sample (0, 1, or 2)

print("\n=== K-Means (k=3) ===")
print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Inertia:       {km.inertia_:.4f}\n")

# ------------------------------------------------------------
# Step 5 — Visualise clusters (PCA projection to 2-D)
# ------------------------------------------------------------
# K-means operates in 4-D space. PCA reduces it to 2-D for plotting
# while preserving as much variance as possible.

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

print("=== PCA ===")
print(f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}  "
      f"PC2={pca.explained_variance_ratio_[1]:.1%}  "
      f"Total={sum(pca.explained_variance_ratio_):.1%}\n")

colors = ["steelblue", "tomato", "seagreen"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 5a: Clusters found by K-means
for cluster in range(3):
    mask = labels == cluster
    axes[0].scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=colors[cluster], label=f"Cluster {cluster}",
        alpha=0.7, edgecolors="k", linewidths=0.3,
    )
centroids_2d = pca.transform(km.cluster_centers_)
axes[0].scatter(
    centroids_2d[:, 0], centroids_2d[:, 1],
    c="black", marker="X", s=120, label="Centroid", zorder=5,
)
axes[0].set_title("K-Means Clusters (PCA projection)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()

# 5b: True species labels for comparison
for species in range(3):
    mask = y == species
    axes[1].scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        c=colors[species], label=iris.target_names[species],
        alpha=0.7, edgecolors="k", linewidths=0.3,
    )
axes[1].set_title("True Species Labels (PCA projection)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()

fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 6 — Evaluate
# ------------------------------------------------------------
sil = silhouette_score(X_scaled, labels)
ari = adjusted_rand_score(y, labels)

print("=== Evaluation ===")
print(f"Silhouette score:    {sil:.4f}  (range −1 to 1, higher = better separated clusters)")
print(f"Adjusted Rand Index: {ari:.4f}  (range  0 to 1, higher = closer to true labels)\n")

# ------------------------------------------------------------
# Step 7 — Effect of K
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k in zip(axes, [2, 3, 5]):
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km_k.fit_predict(X_scaled)
    sil_k = silhouette_score(X_scaled, labels_k)
    for cluster in range(k):
        mask = labels_k == cluster
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            alpha=0.7, edgecolors="k", linewidths=0.3,
        )
    centroids_k = pca.transform(km_k.cluster_centers_)
    ax.scatter(centroids_k[:, 0], centroids_k[:, 1], c="black", marker="X", s=120, zorder=5)
    ax.set_title(f"K={k}  |  silhouette={sil_k:.3f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

fig.suptitle("Effect of K on Clustering")
fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 8 — Make a single prediction (assign new sample)
# ------------------------------------------------------------
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # new, unseen flower
sample_scaled = scaler.transform(sample)
cluster_id = km.predict(sample_scaled)[0]
distances = km.transform(sample_scaled)[0]  # distance to each centroid

print("=== Single Prediction ===")
print("Input: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
print(f"Assigned to cluster: {cluster_id}")
print("Distances to all centroids:")
for i, d in enumerate(distances):
    marker = " ← assigned" if i == cluster_id else ""
    print(f"  Centroid {i}: {d:.4f}{marker}")
