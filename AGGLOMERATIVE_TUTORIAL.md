# Agglomerative Clustering with scikit-learn — Step-by-Step Tutorial

## Introduction to Agglomerative Clustering

**Agglomerative clustering** is a hierarchical, unsupervised algorithm. Unlike K-means, it does not require the number of clusters to be specified upfront — the full merge history is captured in a **dendrogram**, and you choose how many clusters to extract afterwards.

### How it works

The algorithm works bottom-up:

1. **Start** — every sample is its own cluster (N clusters total)
2. **Merge** — find the two closest clusters and merge them into one
3. **Repeat** — keep merging until all samples form a single cluster

The result is a tree of merges — the dendrogram — which records which clusters merged and at what distance. Cutting the tree at a chosen height gives the final cluster assignments.

### Linkage criteria

"Distance between two clusters" is ambiguous when clusters contain many points. The **linkage criterion** defines how it is measured:

| Linkage | Distance measured as |
|---|---|
| **ward** | Increase in total within-cluster variance after merging (default) |
| **complete** | Distance between the two **farthest** points across clusters |
| **average** | Average distance between **all pairs** of points across clusters |
| **single** | Distance between the two **closest** points across clusters |

Ward linkage tends to produce compact, evenly sized clusters and is the most commonly used.

### K-means vs. Agglomerative

| | K-Means | Agglomerative |
|---|---|---|
| K required upfront | Yes | No — read from dendrogram |
| Cluster shape assumed | Spherical | Flexible |
| Result | Centroids | Dendrogram + labels |
| Scalability | Fast on large data | Slower (O(n²) memory) |
| New sample prediction | `predict()` built in | Requires extra step |

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| No need to specify K in advance | Computationally expensive for large datasets |
| Dendrogram reveals full cluster structure | Cannot undo a merge — greedy algorithm |
| Flexible linkage criteria | Sensitive to outliers (especially single linkage) |
| Deterministic — no random initialisation | No built-in `predict()` for new samples |

---

## Preparation — Environment Setup

Before running any code, install Python and set up an isolated environment.

**Install Python 3.9 or newer** from [python.org](https://www.python.org/downloads/). Verify it is available in your terminal:

```bash
python3 --version
```

Then set up an isolated environment:

```bash
# 1. Create and enter your project folder
mkdir agglomerative && cd agglomerative

# 2. Create a virtual environment (run once)
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install numpy matplotlib scikit-learn scipy

# 5. When you're done, deactivate
deactivate
```

---

## Preparation — Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from scipy.cluster.hierarchy import dendrogram, linkage
```

- **numpy** — numerical operations
- **matplotlib** — plotting
- **sklearn.datasets** — synthetic data generator
- **sklearn.preprocessing** — feature scaling
- **sklearn.cluster** — agglomerative clustering model
- **sklearn.neighbors** — NearestCentroid for assigning new samples
- **scipy.cluster.hierarchy** — dendrogram computation and plotting

---

## Step 1 — Generate & Explore the Data

Generate a synthetic dataset using `make_blobs` with 300 samples, 3 clusters, `cluster_std=0.8`, and `random_state=42`. Print the number of samples and features, and plot the raw data without labels.

<details>
<summary>Solution</summary>

```python
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
```

Using the same dataset as the K-means tutorial makes it easy to compare both algorithms on identical data. The raw scatter plot is shown without labels to simulate the real unsupervised scenario.

</details>

---

## Step 2 — Scale the Features

Fit a `StandardScaler` on `X` and transform it. Print the mean and standard deviation of the first feature before and after scaling.

<details>
<summary>Solution</summary>

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== Feature Scaling ===")
print(f"Before — mean: {X[:, 0].mean():.3f}  std: {X[:, 0].std():.3f}")
print(f"After  — mean: {X_scaled[:, 0].mean():.3f}  std: {X_scaled[:, 0].std():.3f}\n")
```

Agglomerative clustering uses Euclidean distance by default, so features with larger numeric ranges would dominate the distance calculation. Standardising puts all features on equal footing — the same reason as in K-means.

</details>

---

## Step 3 — Visualise the Dendrogram

Compute the linkage matrix using Ward's method and plot the dendrogram. Use `truncate_mode="lastp"` with `p=20` to show only the last 20 merges — the full dendrogram with 300 leaves is unreadable. In 2–3 sentences, explain how to read the dendrogram and what it tells you about the number of clusters.

<details>
<summary>Solution</summary>

```python
Z = linkage(X_scaled, method="ward")

fig, ax = plt.subplots(figsize=(12, 5))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=20, leaf_rotation=45)
ax.set_title("Dendrogram (Ward linkage) — last 20 merges")
ax.set_xlabel("Cluster (sample count)")
ax.set_ylabel("Merge distance")
fig.tight_layout()
plt.show()
```

The **y-axis** is the distance at which two clusters were merged — higher means the merged clusters were farther apart. Reading from the bottom up, small merges happen early (close-by points joining) and large merges happen late (distant groups combining).

The key signal is a **large vertical gap**: a tall unbroken vertical line means the algorithm had to jump a large distance to make that merge — suggesting it was joining two genuinely separate groups. The number of lines you can draw horizontally across the largest gap tells you the natural number of clusters.

For this dataset you should see a clear large gap just before the final merge that combines everything into one group, confirming **3 clusters**.

</details>

---

## Step 4 — Fit Agglomerative Clustering

Fit `AgglomerativeClustering` with `n_clusters=3` and `linkage="ward"`. Print the cluster sizes.

<details>
<summary>Solution</summary>

```python
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = agg.fit_predict(X_scaled)

print("=== Agglomerative Clustering (n_clusters=3, ward) ===")
print(f"Cluster sizes: {np.bincount(labels)}\n")
```

Unlike K-means, agglomerative clustering is **deterministic** — it produces the same result every time because there is no random initialisation. The algorithm always makes the same sequence of greedy merges, so `random_state` is not needed.

`fit_predict` fits the model and returns the cluster labels in one step. Note that there is no separate `predict()` method — the model can only assign labels to the data it was fitted on. Assigning new samples requires an extra step covered in Step 7.

</details>

---

## Step 5 — Visualise the Clusters

Produce a side-by-side figure: agglomerative cluster assignments on the left, and on the right the origin assigned by `make_blobs` (only available because we generated the data — in a real unsupervised problem this plot would not exist). In 1–2 sentences, compare the result to the K-means clusters from the other tutorial.

<details>
<summary>Solution</summary>

```python
colors = ["steelblue", "tomato", "seagreen"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: agglomerative clusters
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

# Right: true labels
for cluster in range(3):
    mask = y == cluster
    axes[1].scatter(
        X_scaled[mask, 0], X_scaled[mask, 1],
        c=colors[cluster], label=f"True cluster {cluster}",
        alpha=0.7, edgecolors="k", linewidths=0.3,
    )
axes[1].set_title("Ground Truth (known only because we generated the data)")
axes[1].set_xlabel("Feature 1 (scaled)")
axes[1].set_ylabel("Feature 2 (scaled)")
axes[1].legend()

fig.tight_layout()
plt.show()
```

**Note on colors:** the cluster indices (0, 1, 2) assigned by the algorithm are arbitrary — there is no reason cluster 0 should correspond to true label 0. Do not compare colors between the two plots. Compare the **geometric groupings** instead: does each blob of points stay together as one color in the left plot? If yes, the algorithm found the correct structure regardless of which color it used.

On this well-separated dataset, agglomerative clustering with Ward linkage recovers the true structure just as cleanly as K-means. The two algorithms produce nearly identical results here because the clusters are spherical and evenly sized — exactly the conditions both methods handle well.

</details>

---

## Step 6 — Effect of Linkage Criteria

Fit agglomerative clustering with all four linkage methods (`ward`, `complete`, `average`, `single`) on the **`make_moons` dataset** — two interleaved crescent shapes. This dataset is non-convex, which means the differences between linkage methods become clearly visible. Produce a 1×4 plot. In 2–3 sentences, explain which linkage method handles the crescent shapes correctly and why.

<details>
<summary>Solution</summary>

```python
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
```

We use `make_moons` here instead of `make_blobs` because well-separated spherical blobs look identical across all linkage methods — there is no ambiguity and no visible difference. Crescent shapes are non-convex, which exposes the fundamental difference in how each method defines "distance between clusters."

**Single linkage** correctly follows the crescent shape — it measures distance by the two *closest* points between clusters, so it can chain along a curved boundary. **Ward**, **complete**, and **average** all assume compact, roughly spherical clusters and cut the crescents at the wrong place, producing one cluster with half of each crescent mixed together.

This is the key insight of linkage criteria: **no single method is universally best**. Ward is the right default for compact blobs; single linkage is the right choice for elongated or non-convex shapes.

</details>

---

## Step 7 — Assign a New Sample

`AgglomerativeClustering` has no `predict()` method — it can only label the data it was fitted on. To assign a new point, fit a `NearestCentroid` classifier on the cluster labels and use it to predict. Assign the point (0.0, 4.0), scale it with the existing scaler, and print the assigned cluster and distance to every centroid.

<details>
<summary>Solution</summary>

```python
from sklearn.neighbors import NearestCentroid

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
```

`NearestCentroid` computes the mean position of each cluster (its centroid) and assigns any new point to the nearest one — exactly the same logic K-means uses internally for prediction. The centroids are computed from the agglomerative cluster labels, not from K-means, so they reflect the hierarchical clustering result.

The new sample must be scaled with `scaler.transform` — not `scaler.fit_transform` — so that the same training statistics are applied.

</details>
