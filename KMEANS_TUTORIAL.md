# K-Means Clustering with scikit-learn — Step-by-Step Tutorial

## Introduction to K-Means Clustering

**K-means** is an unsupervised machine learning algorithm that partitions a dataset into K clusters. Unlike the supervised algorithms in this series (decision trees, logistic regression), K-means receives no labels during training — it discovers structure in the data on its own.

### How it works

The algorithm starts by placing K centroids at random positions. It then alternates between two steps until the assignments no longer change:

1. **Assign** — each sample is assigned to the nearest centroid (by Euclidean distance)
2. **Update** — each centroid is recomputed as the mean of all samples currently assigned to it

This is guaranteed to converge, though not necessarily to the global optimum. Running the algorithm multiple times with different random initialisations (`n_init`) and keeping the best result guards against poor local optima.

### Inertia

The quality of a solution is measured by **inertia** — the sum of squared distances from each sample to its assigned centroid:

```
inertia = Σ ||xᵢ − cₖ||²
```

Lower inertia means samples are more tightly packed around their centroids. Adding more clusters always reduces inertia, which is why inertia alone cannot determine the right K.

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Simple and fast | Requires K to be specified upfront |
| Scales well to large datasets | Sensitive to feature scale — always standardise |
| Easy to interpret cluster centroids | Assumes spherical, equally sized clusters |
| No labels required | Converges to local optima — use `n_init > 1` |

### Supervised vs. unsupervised

| | Supervised (Decision Tree, Logistic Regression) | Unsupervised (K-Means) |
|---|---|---|
| Training data | Features + labels | Features only |
| Goal | Predict a label | Discover groups |
| Evaluation | Accuracy, AUC | Silhouette score, ARI |

---

## Preparation — Environment Setup

Before running any code, install Python and set up an isolated environment.

**Install Python 3.9 or newer** from [python.org](https://www.python.org/downloads/). Verify it is available in your terminal:

```bash
python3 --version
```

> **Windows users:** during installation, tick **"Add Python to PATH"** so the `python` and `pip` commands are available in your terminal.

Then set up an isolated environment:

```bash
# 1. Create and enter your project folder
mkdir kmeans && cd kmeans

# 2. Create a virtual environment (run once)
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install numpy matplotlib scikit-learn

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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

- **numpy** — numerical operations and array handling
- **matplotlib** — plotting and visualisation
- **sklearn.datasets** — synthetic data generator (`make_blobs`)
- **sklearn.preprocessing** — feature scaling
- **sklearn.cluster** — the K-means model
- **sklearn.metrics** — silhouette score and Adjusted Rand Index

---

## Step 1 — Generate & Explore the Data

Use `make_blobs` to generate 300 samples with 3 clusters, a cluster standard deviation of 0.8, and `random_state=42`. Print the number of samples, features, and true cluster sizes. Then plot the raw data as a scatter plot **without** revealing the true labels.

<details>
<summary>Solution</summary>

```python
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
# X: shape (300, 2) — two features, ready to plot directly
# y: true cluster labels (0, 1, 2) — hidden during fitting, used for evaluation

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

**`make_blobs`** generates isotropic Gaussian clusters — each cluster is a spherical cloud of points drawn from a normal distribution centred at a randomly placed centroid. This matches K-means' assumptions well, making it an ideal dataset for learning the algorithm.

**`cluster_std=0.8`** controls the spread of each cluster. Lower values produce tighter, more obviously separated blobs; higher values produce overlap that makes clustering harder.

The raw scatter plot is plotted **without labels** to simulate the real unsupervised scenario — the algorithm must discover the groups from geometry alone.

</details>

---

## Step 2 — Scale the Features

Fit a `StandardScaler` on `X` and transform it. Print the mean and standard deviation of the first feature before and after scaling. In 1–2 sentences, explain why scaling is especially important for K-means.

<details>
<summary>Solution</summary>

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== Feature Scaling ===")
print(f"Before — mean: {X[:, 0].mean():.3f}  std: {X[:, 0].std():.3f}")
print(f"After  — mean: {X_scaled[:, 0].mean():.3f}  std: {X_scaled[:, 0].std():.3f}\n")
```

K-means assigns each sample to the nearest centroid using **Euclidean distance**. A feature measured in a larger numeric range dominates the distance calculation simply because its values are numerically larger — not because it is more informative. Standardising all features to zero mean and unit variance puts them on an equal footing.

For `make_blobs` both features happen to be on a similar scale, so the effect is modest here. In real datasets (e.g. age vs. income) the difference is dramatic and skipping this step would produce meaningless clusters.

</details>

---

## Step 3 — Choose K: Elbow + Silhouette

Fit K-means for K from 2 to 10. For each K record both the inertia and the silhouette score, print the results, and plot both as a side-by-side figure. Identify the best K from the silhouette score. In 2–3 sentences, explain what each method shows and how they complement each other.

<details>
<summary>Solution</summary>

```python
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
```

The **elbow method** plots inertia against K and looks for the point where the rate of decrease sharply slows — the "elbow". It provides good visual intuition but requires a subjective judgement call, and the bend is often gradual rather than sharp in real data.

The **silhouette score** for a sample *i* is:

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

Where `a(i)` is the mean distance to other points in the same cluster and `b(i)` is the mean distance to points in the nearest other cluster. A score near **1** means the sample sits deep inside its cluster and far from others; near **0** means it sits on a boundary; **negative** means it was likely assigned to the wrong cluster. The overall score averages this across all samples.

The two methods complement each other: the elbow gives intuition about diminishing returns from adding clusters, while the silhouette provides an objective maximum — just pick the highest value. With `make_blobs` both methods agree clearly on **K=3**: the elbow shows a sharp drop in inertia at K=3 followed by near-flat gains, and the silhouette peaks at K=3.

</details>

---

## Step 4 — Fit K-Means with K=3

Fit a `KMeans` model with 3 clusters, `random_state=42`, and `n_init=10`. Print the cluster sizes and the final inertia.

<details>
<summary>Solution</summary>

```python
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_scaled)
labels = km.labels_

print("=== K-Means (k=3) ===")
print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Inertia:       {km.inertia_:.4f}\n")
```

**`n_init=10`** runs the algorithm 10 times with different random centroid initialisations and keeps the solution with the lowest inertia. This guards against converging to a poor local optimum — K-means is not convex and different starting points can produce different results.

**`km.labels_`** contains the cluster index (0, 1, or 2) assigned to each of the 300 samples. Cluster indices are arbitrary — cluster 0 does not necessarily correspond to the first true cluster.

</details>

---

## Step 5 — Visualise the Clusters

Because `make_blobs` generates 2-D data, you can plot the clusters directly — no PCA needed. Produce a side-by-side figure:

- **Left** — scatter plot coloured by K-means cluster label, with centroids marked
- **Right** — same data coloured by true label

In 1–2 sentences, compare the two plots.

<details>
<summary>Solution</summary>

```python
colors = ["steelblue", "tomato", "seagreen"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: K-means clusters
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

# Right: true labels
for cluster in range(3):
    mask = y == cluster
    axes[1].scatter(
        X_scaled[mask, 0], X_scaled[mask, 1],
        c=colors[cluster], label=f"True cluster {cluster}",
        alpha=0.7, edgecolors="k", linewidths=0.3,
    )
axes[1].set_title("True Cluster Labels")
axes[1].set_xlabel("Feature 1 (scaled)")
axes[1].set_ylabel("Feature 2 (scaled)")
axes[1].legend()

fig.tight_layout()
plt.show()
```

The two plots should look nearly identical — K-means recovers the true cluster structure almost perfectly because `make_blobs` generates well-separated spherical clusters that match K-means' assumptions. The centroids (black X markers) sit at the geometric centre of each cluster.

</details>

---

## Step 6 — Effect of K

Fit K-means with K=2, K=3, and K=5. For each, produce a scatter plot with the silhouette score in the title. In 2–3 sentences, describe what happens to the clusters and the silhouette score as K moves away from 3.

<details>
<summary>Solution</summary>

```python
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
```

With **K=2** the algorithm merges two of the true clusters into one — the silhouette score drops because many points end up far from their centroid. With **K=5** the algorithm splits natural clusters in half, creating artificial boundaries through dense regions — the silhouette score also drops because points that belong together are separated. **K=3** achieves the highest score by matching the true structure of the data.

</details>

---

## Step 7 — Make a Single Prediction (Assign New Sample)

Assign a new data point at coordinates (0.0, 4.0) to a cluster. Remember to scale the new point using the same scaler fitted on the training data. Print the assigned cluster and the distance to every centroid.

<details>
<summary>Solution</summary>

```python
sample = np.array([[0.0, 4.0]])
sample_scaled = scaler.transform(sample)
cluster_id = km.predict(sample_scaled)[0]
distances  = km.transform(sample_scaled)[0]

print("=== Single Prediction ===")
print(f"Input: Feature1=0.0, Feature2=4.0")
print(f"Assigned to cluster: {cluster_id}")
print("Distances to all centroids:")
for i, d in enumerate(distances):
    marker = " ← assigned" if i == cluster_id else ""
    print(f"  Centroid {i}: {d:.4f}{marker}")
```

The new point must be scaled with `scaler.transform` — not `scaler.fit_transform` — so that it is shifted and scaled by the same training statistics. Re-fitting on a single point would produce meaningless statistics.

`km.transform` returns the distance from the new point to every centroid. The model assigns it to the nearest one. These distances also serve as a confidence measure: a point very close to one centroid and far from all others is a clear cluster member; a point roughly equidistant from two centroids sits on a boundary and its assignment is less certain.

</details>
