"""
Workshop Tutorial: Agglomerative Clustering with scikit-learn
=============================================================

Agglomerative clustering is a hierarchical, unsupervised algorithm.
Unlike K-means, it does not require K to be specified upfront.

It works bottom-up:
  1. Start: every sample is its own cluster (N clusters)
  2. Repeatedly merge the two closest clusters
  3. Stop when all samples form a single cluster

The full merge history is captured in a dendrogram — a tree diagram
that shows which clusters merged at what distance. Cutting the
dendrogram at a chosen height gives the final cluster assignments.

We use the same make_blobs dataset as the K-means tutorial so
both algorithms can be compared on identical data.
"""

# ------------------------------------------------------------
# Step 0 — Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from scipy.cluster.hierarchy import dendrogram, linkage

# ------------------------------------------------------------
# Step 1 — Generate & explore the data
# ------------------------------------------------------------
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"True cluster sizes: {np.bincount(y)}\n")

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
# Agglomerative clustering uses Euclidean distance by default —
# the same reason as K-means to standardise before fitting.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== Feature Scaling ===")
print(f"Before — mean: {X[:, 0].mean():.3f}  std: {X[:, 0].std():.3f}")
print(f"After  — mean: {X_scaled[:, 0].mean():.3f}  std: {X_scaled[:, 0].std():.3f}\n")

# ------------------------------------------------------------
# Step 3 — Visualise the dendrogram
# ------------------------------------------------------------
# The dendrogram shows the full merge history. The y-axis is the
# distance at which two clusters merged. A large vertical gap
# before the next merge suggests a natural cut point.

Z = linkage(X_scaled, method="ward")

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=20, leaf_rotation=45)
ax.set_title("Dendrogram (Ward linkage) — last 20 merges")
ax.set_xlabel("Cluster (sample count)")
ax.set_ylabel("Merge distance")
fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 4 — Fit agglomerative clustering with n_clusters=3
# ------------------------------------------------------------
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = agg.fit_predict(X_scaled)

print("=== Agglomerative Clustering (n_clusters=3, ward) ===")
print(f"Cluster sizes: {np.bincount(labels)}\n")

# ------------------------------------------------------------
# Step 5 — Visualise clusters
# ------------------------------------------------------------
colors = ["steelblue", "tomato", "seagreen"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 5a: Clusters found by agglomerative clustering
for cluster in range(3):
    mask = labels == cluster
    axes[0].scatter(
        X_scaled[mask, 0], X_scaled[mask, 1],
        c=colors[cluster], label=f"Cluster {cluster}",
        alpha=0.7, edgecolors="k", linewidths=0.3,
    )
axes[0].set_title("Agglomerative Clusters")
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
# Step 6 — Effect of linkage criteria
# ------------------------------------------------------------
# Linkage defines how distance between two clusters is measured:
#   ward     — minimises variance within merged clusters (default, works well)
#   complete — distance between the two farthest points
#   average  — average distance between all pairs of points
#   single   — distance between the two closest points (chaining effect)
#
# make_blobs produces perfectly separated spherical clusters — all linkage
# methods agree there. To actually see the difference, we use make_moons:
# two interleaved crescent shapes that are non-convex. Single linkage
# chains along the crescent correctly; ward and complete cut them into
# circular blobs and get it wrong.

from sklearn.datasets import make_moons
X_moons, _ = make_moons(n_samples=200, noise=0.07, random_state=42)
X_moons_scaled = StandardScaler().fit_transform(X_moons)

linkages = ["ward", "complete", "average", "single"]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, link in zip(axes, linkages):
    agg_l = AgglomerativeClustering(n_clusters=2, linkage=link)
    labels_l = agg_l.fit_predict(X_moons_scaled)
    for cluster in range(2):
        mask = labels_l == cluster
        ax.scatter(
            X_moons_scaled[mask, 0], X_moons_scaled[mask, 1],
            alpha=0.7, edgecolors="k", linewidths=0.3,
        )
    ax.set_title(link)
    ax.set_xlabel("Feature 1 (scaled)")
    ax.set_ylabel("Feature 2 (scaled)")

fig.suptitle("Effect of Linkage Criteria — make_moons dataset")
fig.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 7 — Assign a new sample
# ------------------------------------------------------------
# AgglomerativeClustering has no predict() method — it only assigns
# labels to the data it was fitted on. To assign a new point, fit a
# NearestCentroid classifier on the cluster labels, then use it to
# predict the cluster for any new sample.

nc = NearestCentroid()
nc.fit(X_scaled, labels)

sample = np.array([[0.0, 4.0]])
sample_scaled = scaler.transform(sample)
cluster_id = nc.predict(sample_scaled)[0]

print("=== Assign New Sample ===")
print("Input: Feature1=0.0, Feature2=4.0")
print(f"Assigned to cluster: {cluster_id}")
print("Centroid distances:")
for i, centroid in enumerate(nc.centroids_):
    d = np.linalg.norm(sample_scaled - centroid)
    marker = " ← assigned" if i == cluster_id else ""
    print(f"  Centroid {i}: {d:.4f}{marker}")
