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
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
```

- **numpy** — numerical operations and array handling
- **matplotlib** — plotting and visualisation
- **sklearn.datasets** — built-in datasets (Iris)
- **sklearn.preprocessing** — feature scaling
- **sklearn.cluster** — the K-means model
- **sklearn.decomposition** — PCA for 2-D visualisation
- **sklearn.metrics** — silhouette score and Adjusted Rand Index

---

## Step 1 — Load & Explore the Data

Load the Iris dataset. Assign the feature matrix to `X` and note that `y` (the true species labels) are kept hidden during training — K-means is unsupervised and does not use them. Print the number of samples, the feature names, and the class distribution.

<details>
<summary>Solution</summary>

```python
iris = load_iris()

X = iris.data    # shape: (150, 4) — we cluster on these
y = iris.target  # 0=setosa, 1=versicolor, 2=virginica — hidden during training

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}  ->  {list(iris.feature_names)}")
print(f"True classes (hidden during training): {iris.target_names}")
print(f"Class distribution: {np.bincount(y)}\n")
```

The Iris dataset has 150 samples across 3 species, each described by 4 measurements. Because K-means is unsupervised, `y` is not passed to the model. We only use it later to evaluate how well the discovered clusters align with the true biological species.

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

K-means assigns each sample to the nearest centroid using **Euclidean distance**. A feature measured in larger units (e.g. sepal length in centimetres vs. a feature in millimetres) would dominate the distance calculation simply because its values are numerically larger — not because it is more informative. Standardising all features to zero mean and unit variance puts them on an equal footing.

</details>

---

## Step 3 — Choose K: the Elbow Method

Train K-means models for K from 1 to 10. Record the inertia for each K and plot it. In 2–3 sentences, describe what the elbow method reveals and what K you would choose for the Iris dataset.

<details>
<summary>Solution</summary>

```python
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
```

Inertia always decreases as K increases — with K equal to the number of samples, every point is its own cluster and inertia is zero. The useful signal is the **rate of decrease**: it is steep at low K (adding clusters provides large gains) and flattens at higher K (diminishing returns). The point where the curve bends — the "elbow" — marks a reasonable K. For the Iris dataset the elbow appears at **K=3**, consistent with the three known species.

</details>

---

## Step 4 — Train K-Means with K=3

Fit a `KMeans` model with 3 clusters, `random_state=42`, and `n_init=10`. Print the cluster sizes and the final inertia.

<details>
<summary>Solution</summary>

```python
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_scaled)
labels = km.labels_   # cluster assignment for each sample (0, 1, or 2)

print("\n=== K-Means (k=3) ===")
print(f"Cluster sizes: {np.bincount(labels)}")
print(f"Inertia:       {km.inertia_:.4f}\n")
```

**`n_init=10`** runs the algorithm 10 times with different random centroid initialisations and keeps the solution with the lowest inertia. This guards against the algorithm converging to a poor local optimum.

**`km.labels_`** contains the cluster index (0, 1, or 2) assigned to each of the 150 samples. Note that cluster indices are arbitrary — cluster 0 does not necessarily correspond to species 0 (setosa).

</details>

---

## Step 5 — Visualise the Clusters

K-means operates in 4-D feature space, which cannot be plotted directly. Use PCA to project the scaled data down to 2 components and produce a side-by-side figure:

- **Left** — scatter plot coloured by K-means cluster label, with centroids marked
- **Right** — same projection coloured by true species label

Print the variance explained by each principal component. In 1–2 sentences, compare the two plots.

<details>
<summary>Solution</summary>

```python
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

print("=== PCA ===")
print(f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}  "
      f"PC2={pca.explained_variance_ratio_[1]:.1%}  "
      f"Total={sum(pca.explained_variance_ratio_):.1%}\n")

colors = ["steelblue", "tomato", "seagreen"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: K-means clusters
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

# Right: true species labels
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
```

The two PCA components together capture around **97%** of the total variance, making the 2-D projection a faithful representation of the original 4-D structure. Comparing the two plots shows that K-means recovers the *setosa* cluster perfectly (it is linearly separable) but makes some mistakes on the boundary between *versicolor* and *virginica*, which overlap in feature space.

</details>

---

## Step 6 — Evaluate

Compute two evaluation metrics and print them with a short interpretation:

- **Silhouette score** — measures how tight and well-separated the clusters are, purely from the data geometry (no labels needed)
- **Adjusted Rand Index (ARI)** — compares the cluster assignments to the true labels, correcting for chance

<details>
<summary>Solution</summary>

```python
sil = silhouette_score(X_scaled, labels)
ari = adjusted_rand_score(y, labels)

print("=== Evaluation ===")
print(f"Silhouette score:    {sil:.4f}  (range −1 to 1, higher = better separated clusters)")
print(f"Adjusted Rand Index: {ari:.4f}  (range  0 to 1, higher = closer to true labels)\n")
```

**Silhouette score** for sample *i* is:

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

Where `a(i)` is the mean distance to other points in the same cluster and `b(i)` is the mean distance to points in the nearest other cluster. A score near 1 means the sample is well inside its cluster and far from others; near 0 means it sits on a boundary; negative means it was likely assigned to the wrong cluster. The overall score averages this across all samples.

**Adjusted Rand Index** measures the overlap between two partitions of the same data — here, the K-means labels vs. the true species. It is adjusted for chance so that a random assignment scores near 0, and a perfect match scores 1.0. Unlike silhouette score, ARI requires ground-truth labels and cannot be used in a pure unsupervised setting.

</details>

---

## Step 7 — Effect of K

Train K-means with K=2, K=3, and K=5. For each, produce a scatter plot of the PCA projection and print the silhouette score in the title. In 2–3 sentences, describe what happens to the clusters and the silhouette score as K moves away from 3.

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
```

With **K=2** the algorithm merges *versicolor* and *virginica* into one cluster — it underfits the structure and the silhouette score is lower. With **K=5** it splits the natural groups into artificial sub-clusters that have no biological meaning — inertia drops but the silhouette score also falls because the extra clusters cut through dense regions rather than separating genuinely distinct groups. **K=3** hits the right balance, confirming the elbow method's recommendation.

</details>

---

## Step 8 — Make a Single Prediction (Assign New Sample)

Using the trained K=3 model, assign a new flower sample to a cluster. Use the same measurements as in the decision tree tutorial: sepal length=5.1, sepal width=3.5, petal length=1.4, petal width=0.2. Print the assigned cluster and the distance to every centroid.

<details>
<summary>Solution</summary>

```python
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # new, unseen flower
sample_scaled = scaler.transform(sample)
cluster_id = km.predict(sample_scaled)[0]
distances  = km.transform(sample_scaled)[0]  # distance to each centroid

print("=== Single Prediction ===")
print("Input: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
print(f"Assigned to cluster: {cluster_id}")
print("Distances to all centroids:")
for i, d in enumerate(distances):
    marker = " ← assigned" if i == cluster_id else ""
    print(f"  Centroid {i}: {d:.4f}{marker}")
```

**Important:** the new sample must be scaled with the **same scaler** fitted on the training data — `scaler.transform(sample)`, not `scaler.fit_transform(sample)`. Re-fitting on a single sample would compute nonsensical statistics.

`km.transform` returns the distance from the sample to every centroid. The model assigns the sample to the centroid with the smallest distance. These distances are also useful as a soft confidence measure — a sample very close to one centroid and far from all others is a clear cluster member, while a sample equidistant from two centroids sits on a boundary and its assignment is less certain.

This sample (short petals, narrow petals) is characteristic of *Iris setosa* and should be assigned to the cluster that most closely corresponds to setosa.

</details>
