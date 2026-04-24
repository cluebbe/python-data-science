# Logistic Regression with scikit-learn — Step-by-Step Tutorial

## Introduction to Logistic Regression

**Logistic regression** is a linear classification algorithm that models the probability that a sample belongs to a class using the **sigmoid (logistic) function**. Despite its name it is a *classifier*, not a regressor: it outputs a probability between 0 and 1 and assigns the sample to the class with the higher probability.

### How it works

For binary classification, logistic regression learns a weight vector **w** and bias b that map the input features to a log-odds score, which is then squashed through the sigmoid:

| Step | Formula | Meaning |
|---|---|---|
| Linear combination | z = **w** · **x** + b | Weighted sum of features |
| Sigmoid | p = σ(z) = 1 / (1 + e⁻ᶻ) | Maps z to a probability in [0, 1] |
| Decision | ŷ = 1 if p ≥ 0.5, else 0 | Hard class label |

Training minimises the **binary cross-entropy (log-loss)** over all training samples:

> L = −(1/n) Σ [ yᵢ log(pᵢ) + (1 − yᵢ) log(1 − pᵢ) ]

The decision boundary is a *hyperplane* in feature space. Logistic regression captures global linear separability but cannot model curved or axis-aligned boundaries the way tree-based methods can.

### Regularisation

Without regularisation the model can overfit, especially with many or correlated features. scikit-learn adds an L2 (ridge) penalty by default:

> L_reg = L + (1 / 2C) Σ wⱼ²

The hyperparameter **C** is the *inverse* of regularisation strength:

| C value | Effect |
|---|---|
| Small C | Strong regularisation → weights shrink toward zero → underfitting risk |
| Large C | Weak regularisation → weights can grow freely → overfitting risk |

### Feature scaling is essential

Tree-based methods are invariant to feature scale — they only compare relative values within one feature at a time. Logistic regression is **not**: features with large magnitude dominate the dot product **w** · **x**, making training unstable and coefficients incomparable. Always **standardise** features (zero mean, unit variance) before fitting a logistic regression.

### Key hyperparameters

| Parameter | Role |
|---|---|
| `C` | Inverse regularisation strength — primary tuning lever |
| `penalty` | `"l2"` (default, ridge) or `"l1"` (lasso, produces sparsity) |
| `solver` | Optimisation algorithm — `"lbfgs"` for small/medium datasets |
| `max_iter` | Maximum solver iterations — raise if a `ConvergenceWarning` fires |

### Logistic regression vs. tree-based classifiers

| Property | Logistic Regression | Decision Tree / RF |
|---|---|---|
| Decision boundary | Linear (hyperplane) | Axis-aligned splits (piecewise) |
| Feature scaling | Required | Not needed |
| Probability output | Well-calibrated | Often poorly calibrated |
| Interpretability | Signed coefficients | Feature importances / tree diagram |
| High-dim sparse data | Works well | Slower, prone to overfitting |
| Non-linear patterns | Misses them | Captures them naturally |

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
mkdir logistic-regression && cd logistic-regression

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

> **Why a virtual environment?** It keeps the packages for this project separate from your system Python and other projects, avoiding version conflicts.

## Preparation — Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)
```

- **numpy** — numerical operations and array handling
- **matplotlib** — plotting and visualisation
- **StandardScaler** — zero-mean, unit-variance feature normalisation
- **LogisticRegression** — scikit-learn's logistic regression classifier
- **DecisionTreeClassifier** — used in Step 9 for comparison
- **make_pipeline / PartialDependenceDisplay** — used in Step 12 for partial dependence plots
- **roc_auc_score / RocCurveDisplay** — ROC curve and AUC metric

---

## Step 1 — Load & Explore the Data

Load the Breast Cancer Wisconsin dataset. Assign the feature matrix to `X` and the target vector to `y`. Print the number of samples, feature names, class names, and class distribution.

<details>
<summary>Solution</summary>

```python
bc = load_breast_cancer()

X = bc.data    # shape: (569, 30) — cell nucleus measurements
y = bc.target  # 0=malignant, 1=benign

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Feature names: {bc.feature_names.tolist()}")
print(f"Classes:  {bc.target_names}")
print(f"Class distribution: {np.bincount(y)}  (0=malignant, 1=benign)\n")
```

The **Breast Cancer Wisconsin** dataset contains 569 patient samples described by 30 numerical measurements of cell nuclei extracted from biopsy images. Each base measurement (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) is computed as the mean, standard error, and worst value across all cells — giving 10 × 3 = 30 features in total.

The task is binary classification: **malignant** (212 samples) vs. **benign** (357 samples). The dataset is mildly imbalanced — ~37% malignant, ~63% benign — which is worth keeping in mind when interpreting accuracy alone. It is an ideal dataset for demonstrating logistic regression because the class boundary is largely linear and clinicians benefit from calibrated probability estimates rather than hard labels.

</details>

---

## Step 2 — Split into Train / Test Sets

Split the data into training and test sets using an 80/20 ratio. Ensure the split is reproducible and that class proportions are preserved in both sets.

<details>
<summary>Solution</summary>

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% held out for evaluation
    random_state=42,  # reproducibility
    stratify=y,       # preserve the ~37/63 malignant/benign ratio in both sets
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}\n")
```

- **test_size=0.2** — ~114 samples held out for evaluation; ~455 used for training.
- **random_state=42** — fixes the random seed so results are reproducible across runs.
- **stratify=y** — ensures the malignant/benign proportions are the same in both splits, preventing an accidentally all-benign test set.

</details>

---

## Step 3 — Scale the Features

Fit a `StandardScaler` on the **training data only**, then apply it to both train and test sets. Print the mean and standard deviation of the first feature before and after scaling. In 2–3 sentences, explain why fitting the scaler on the training set only is important.

<details>
<summary>Solution</summary>

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # fit on train, then transform
X_test_sc  = scaler.transform(X_test)       # transform only — never fit on test

print("=== Feature Scaling ===")
print(f"Before scaling — mean: {X_train[:, 0].mean():.3f}  std: {X_train[:, 0].std():.3f}")
print(f"After  scaling — mean: {X_train_sc[:, 0].mean():.3f}  std: {X_train_sc[:, 0].std():.3f}\n")
```

`StandardScaler` subtracts the feature mean and divides by the standard deviation, transforming each feature to zero mean and unit variance. This makes the features directly comparable and prevents those with large raw values (e.g. area, measured in mm²) from dominating the dot product **w** · **x**.

Fitting the scaler on the training set only prevents **data leakage**: if the test set were included when computing the mean and standard deviation, the model would indirectly see test distribution statistics during training, inflating apparent performance. The test set must be treated as entirely unseen — it is transformed using the statistics derived from training data only.

</details>

---

## Step 4 — Train a Logistic Regression Classifier

Fit a `LogisticRegression` with `C=1.0` and `max_iter=10000` on the scaled training data. Print the training accuracy and the number of iterations the solver actually used.

<details>
<summary>Solution</summary>

```python
lr = LogisticRegression(C=1.0, max_iter=10000, random_state=42)
lr.fit(X_train_sc, y_train)

train_acc = accuracy_score(y_train, lr.predict(X_train_sc))
print("=== Training ===")
print(f"Training accuracy: {train_acc:.2%}")
print(f"Solver iterations: {lr.n_iter_[0]}\n")
```

Key hyperparameters:

| Parameter | Effect |
|---|---|
| `C` | Inverse regularisation strength. `C=1` is the scikit-learn default and a reasonable starting point; tune via cross-validation. |
| `max_iter` | The solver (lbfgs by default) is iterative. The default of 100 often triggers a `ConvergenceWarning` on high-dimensional data; 10 000 is generous. |
| `n_iter_` | Reports how many iterations were actually used. If it equals `max_iter`, the model did not converge — raise the limit or rescale features more aggressively. |

</details>

---

## Step 5 — Evaluate the Model

Generate predictions on the test set. Print the test accuracy, the ROC-AUC score, and a full classification report. Also print the predicted class and probability for the first five test samples. In 2–3 sentences, explain what ROC-AUC measures and why it can be more informative than accuracy for imbalanced datasets.

<details>
<summary>Solution</summary>

```python
y_pred = lr.predict(X_test_sc)
y_prob = lr.predict_proba(X_test_sc)[:, 1]  # P(benign) for each test sample

test_acc = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_prob)

print("=== Evaluation ===")
print(f"Test accuracy: {test_acc:.2%}")
print(f"ROC-AUC:       {roc_auc:.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=bc.target_names))

print("Predicted probabilities (first 5 test samples):")
for i in range(5):
    print(f"  sample {i}: p(benign)={y_prob[i]:.3f}  "
          f"predicted={bc.target_names[y_pred[i]]}  "
          f"true={bc.target_names[y_test[i]]}")
print()
```

**ROC-AUC** (Area Under the Receiver Operating Characteristic Curve) measures the probability that the model ranks a randomly chosen benign sample above a randomly chosen malignant one, across *all* possible classification thresholds. An AUC of 1.0 means perfect separation; 0.5 is no better than random.

AUC is preferable to accuracy on imbalanced datasets because it is unaffected by the class ratio — a naïve classifier that always predicts "benign" achieves 63% accuracy on this dataset but an AUC of exactly 0.5. For medical applications, AUC is also useful for choosing an operating threshold: lowering the threshold below 0.5 increases recall on the malignant class (fewer missed cancers) at the cost of more false alarms, and the ROC curve makes this trade-off explicit.

</details>

---

## Step 6 — Understand the Coefficients

Extract the model's coefficient vector (`lr.coef_[0]`) and print all features sorted by absolute coefficient magnitude. Include the sign direction for each. In 2–3 sentences, explain what a large positive or large negative coefficient means for the predicted probability, and why standardising the features first is necessary for coefficients to be comparable.

<details>
<summary>Solution</summary>

```python
coefs = lr.coef_[0]          # shape (30,) — one coefficient per feature
sorted_idx = np.argsort(np.abs(coefs))[::-1]

print("=== Model Coefficients (sorted by |coef|) ===")
for rank, idx in enumerate(sorted_idx, 1):
    direction = "→ benign" if coefs[idx] > 0 else "→ malignant"
    print(f"  {rank:2}. {bc.feature_names[idx]:<35} coef={coefs[idx]:+.4f}  {direction}")
print()
```

Because features are standardised, coefficients are directly comparable in magnitude: a coefficient of +2 means a one-standard-deviation increase in that feature raises the log-odds of being benign by 2, roughly tripling the benign odds. A large negative coefficient points strongly toward malignancy.

Unlike tree-based feature importances — which only report magnitude — logistic regression coefficients carry **sign**: they tell you not just which features matter but in which direction they push the prediction. Features with coefficients near zero have little marginal influence and could potentially be removed without hurting performance.

</details>

---

## Step 7 — Visualise

Display the following three plots. Do not save figures to disk — display them inline.

- **7a** — A confusion matrix heatmap for the test set.
- **7b** — A ROC curve with the AUC score annotated, including the random-classifier diagonal for reference.
- **7c** — A horizontal bar chart of the top 15 coefficients by absolute magnitude, coloured blue for positive (→ benign) and red for negative (→ malignant).

<details>
<summary>Solution</summary>

### 7a: Confusion Matrix

```python
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=bc.target_names,
    cmap="Blues",
    ax=ax,
)
ax.set_title("Confusion Matrix — Logistic Regression")
fig.tight_layout()
plt.show()
```

For cancer classification, the most dangerous cell is the **top-right** (false negatives — malignant predicted as benign). Missing a malignant tumour is far more costly than a false alarm. If this cell is non-zero, consider lowering the classification threshold below 0.5 to improve malignant recall at the expense of benign precision.

### 7b: ROC Curve

```python
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(lr, X_test_sc, y_test, ax=ax, name="Logistic Regression")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
ax.set_title(f"ROC Curve — Logistic Regression (AUC = {roc_auc:.3f})")
ax.legend()
fig.tight_layout()
plt.show()
```

The ROC curve plots True Positive Rate (recall on the positive class) against False Positive Rate across all thresholds. A perfect classifier hugs the top-left corner; a random classifier follows the diagonal. The area under the curve summarises discrimination ability in a single number.

### 7c: Coefficient Bar Chart

```python
top_n = 15
top_idx = sorted_idx[:top_n]
colors = ["steelblue" if coefs[i] > 0 else "tomato" for i in top_idx]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(
    range(top_n),
    coefs[top_idx][::-1],
    color=colors[::-1],
)
ax.set_yticks(range(top_n))
ax.set_yticklabels([bc.feature_names[i] for i in top_idx][::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient (standardised features)")
ax.set_title(f"Top {top_n} Logistic Regression Coefficients")
fig.tight_layout()
plt.show()
```

Colour-coding by direction makes it immediately clear which features pull the prediction toward benign (blue, right of zero) and which toward malignant (red, left of zero). Because features are standardised, bar lengths are directly comparable — a longer bar has a stronger marginal effect per standard deviation.

</details>

---

## Step 8 — Regularisation: C vs. Accuracy

Train logistic regression models with `C` ∈ {0.001, 0.01, 0.1, 1, 10, 100}. Record train and test accuracy for each value and plot both curves on a log-scale x-axis. In 2–3 sentences, explain what the curves reveal about underfitting and overfitting, and when you should increase or decrease C.

<details>
<summary>Solution</summary>

```python
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores, test_scores = [], []

print("=== Regularisation Strength (C) vs Accuracy ===")
for C in C_values:
    m = LogisticRegression(C=C, max_iter=10000, random_state=42)
    m.fit(X_train_sc, y_train)
    tr = accuracy_score(y_train, m.predict(X_train_sc))
    te = accuracy_score(y_test,  m.predict(X_test_sc))
    train_scores.append(tr)
    test_scores.append(te)
    print(f"  C={C:<8}  train={tr:.2%}  test={te:.2%}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(C_values, train_scores, marker="o", label="Train accuracy")
ax.plot(C_values, test_scores,  marker="s", label="Test accuracy")
ax.set_xscale("log")
ax.set_xlabel("C (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("Regularisation Strength vs Accuracy — Logistic Regression")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
plt.show()
```

At very small C (strong regularisation) both train and test accuracy are depressed — the weights are shrunk so aggressively toward zero that the model cannot fit even the training data, a sign of **underfitting**. As C increases the model gains flexibility and accuracy rises for both sets. If train accuracy continues to climb while test accuracy plateaus or falls, the model is **overfitting** — that widening gap is the signal to reduce C and strengthen regularisation.

For this dataset C=1 sits near the sweet spot: the Breast Cancer features are well-scaled and largely linearly separable, so heavy regularisation is not needed. In practice, choose C via cross-validation (e.g. `LogisticRegressionCV`) rather than by manual inspection of the train/test curve.

</details>

---

## Step 9 — Logistic Regression vs. Decision Tree

Train a `DecisionTreeClassifier` with `max_depth=3` and `random_state=42` on the *unscaled* training data (trees do not need scaling). Compare its test accuracy and ROC-AUC against the logistic regression model trained in Step 4. In 2–3 sentences, explain the result and describe a scenario where the decision tree would be expected to outperform logistic regression.

<details>
<summary>Solution</summary>

```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)    # unscaled — trees are scale-invariant
dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
lr_acc = accuracy_score(y_test, y_pred)
dt_auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])

print("=== Logistic Regression vs. Decision Tree ===")
print(f"  Logistic Regression — accuracy: {lr_acc:.2%}  AUC: {roc_auc:.4f}")
print(f"  Decision Tree       — accuracy: {dt_acc:.2%}  AUC: {dt_auc:.4f}")
print(f"  Accuracy gap: {lr_acc - dt_acc:+.2%}\n")
```

On this dataset the classes are largely linearly separable — logistic regression finds a single hyperplane that cleanly divides most malignant from benign samples, while a depth-3 tree can only approximate that boundary with three axis-aligned cuts. The AUC gap is often more revealing than accuracy: a higher AUC means the logistic model's probability scores rank samples more reliably across all thresholds.

For problems with strongly non-linear class boundaries — for example, when the positive class occupies an irregular region of feature space — a decision tree (or random forest) would be expected to close or reverse this gap, since its piecewise boundary can adapt to arbitrary shapes that no single hyperplane can capture.

</details>

---

## Step 10 — Make a Single Prediction (Inference)

Using the trained logistic regression, predict the class and class probabilities for the first sample in the dataset. Remember to scale the sample before predicting. Print all 30 input feature values, the predicted class, the true label, and the probability for each class. In 2–3 sentences, explain why logistic regression's probability output is described as "well-calibrated" and why this matters clinically.

<details>
<summary>Solution</summary>

```python
sample_raw = bc.data[[0]]                  # first sample — known malignant (label=0)
sample_sc  = scaler.transform(sample_raw)  # scale with the already-fitted scaler
prediction    = lr.predict(sample_sc)[0]
probabilities = lr.predict_proba(sample_sc)[0]

print("=== Single Prediction ===")
print("Input features (first sample from dataset):")
for name, val in zip(bc.feature_names, sample_raw[0]):
    print(f"  {name:<35} {val:.4f}")
print(f"\nPredicted class: {bc.target_names[prediction]}")
print(f"True label:      {bc.target_names[bc.target[0]]}")
print("Class probabilities:")
for cls, prob in zip(bc.target_names, probabilities):
    print(f"  {cls:<12} {prob:.2%}")
```

- `predict` returns the class with the higher sigmoid probability (equivalently, the class on the correct side of the decision hyperplane).
- `predict_proba` returns the sigmoid output — a number in [0, 1] derived from the linear score, not from a leaf's training-sample count.

Unlike a decision tree — which assigns a hard probability based on the fraction of training samples in a leaf — logistic regression's sigmoid output is **well-calibrated**: a predicted P(malignant) = 0.97 genuinely reflects ~97% confidence. In clinical decision support this enables evidence-based threshold setting: a clinician might flag any sample with P(malignant) > 0.20 for biopsy, accepting more false alarms to catch nearly every true positive, rather than being forced to use the fixed 0.5 cutoff.

</details>

---

## Step 11 — DPL: Difference in Proportions of Labels

### Background

**DPL** (Difference in Proportions of Labels) is a fairness metric that measures whether two subgroups have different rates of positive outcomes — first in the ground-truth data, and then in the model's predictions:

> DPL = P(positive label | group A) − P(positive label | group B)

A DPL near zero means the two groups have similar label rates. A large absolute DPL flags a disparity: either the data itself contains a structural imbalance between groups, or the model amplifies that imbalance beyond what the labels alone justify.

### Task

The Breast Cancer dataset has no demographic features, so construct a binary group by splitting on the **median value of `mean_area`** (feature index 3): samples below the median form the *small-tumour* group; samples at or above the median form the *large-tumour* group.

1. Compute DPL for the ground-truth benign labels in the training set.
2. Compute DPL for the model's predicted labels on the test set.
3. Plot both as side-by-side grouped bar charts.
4. In 2–3 sentences, interpret the results — does the model amplify or mitigate the data-level disparity?

<details>
<summary>Solution</summary>

```python
area_median = np.median(X_train[:, 3])
grp_train   = (X_train[:, 3] >= area_median).astype(int)   # 1=large, 0=small
grp_test    = (X_test[:, 3]  >= area_median).astype(int)

p_label_large = y_train[grp_train == 1].mean()
p_label_small = y_train[grp_train == 0].mean()
dpl_labels    = p_label_large - p_label_small

p_pred_large  = y_pred[grp_test == 1].mean()
p_pred_small  = y_pred[grp_test == 0].mean()
dpl_preds     = p_pred_large - p_pred_small

print("=== DPL: Difference in Proportions of Labels ===")
print(f"  Group split: mean_area median = {area_median:.1f}")
print(f"  Training set — P(benign | large): {p_label_large:.3f}   "
      f"P(benign | small): {p_label_small:.3f}   DPL = {dpl_labels:+.3f}")
print(f"  Test set     — P(ŷ=benign | large): {p_pred_large:.3f}   "
      f"P(ŷ=benign | small): {p_pred_small:.3f}   DPL = {dpl_preds:+.3f}\n")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (title, vals) in zip(axes, [
    ("Ground-truth labels (train)", [p_label_small, p_label_large]),
    ("Model predictions (test)",    [p_pred_small,  p_pred_large]),
]):
    bars = ax.bar(
        ["small tumour\n(below median)", "large tumour\n(above median)"],
        vals, color=["steelblue", "tomato"], width=0.5,
    )
    ax.bar_label(bars, fmt="{:.3f}", padding=3)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("P(benign)")
    ax.set_title(title)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
fig.suptitle("DPL: Proportion of Benign by Tumour-Size Group")
fig.tight_layout()
plt.show()
```

A negative DPL confirms a clinically expected pattern: large-tumour samples are substantially less frequently benign than small-tumour samples, both in the data and in the model's predictions. If the model's |DPL| exceeds the data's |DPL|, the model is **amplifying** the disparity — producing a stronger group bias than the labels alone warrant, which would merit further investigation. If the model's |DPL| is smaller, it is partially **mitigating** the data-level disparity.

In real applications, DPL would be computed across demographic groups (age, sex, ethnicity) rather than a clinical feature, and a threshold — often |DPL| < 0.1 — is used to decide whether the disparity is acceptable.

</details>

---

## Step 12 — Partial Dependence Plots (PDPs)

### Background

A **Partial Dependence Plot (PDP)** shows the *marginal* effect of one feature on the predicted probability, averaged over the joint distribution of all other features. For each grid value of feature j, the model predicts every training sample with xⱼ fixed at that grid value; the average prediction is plotted:

> PDP(xⱼ) = (1/n) Σᵢ f(x₁ⁱ, …, xⱼ, …, xₙⁱ)

The resulting curve answers: *"if I could dial this feature up or down while everything else stayed realistic, how would the predicted probability change?"* Unlike a coefficient (which captures the effect in log-odds space), a PDP shows the effect in probability space, making it more directly interpretable.

### Task

Wrap the scaler and classifier in a `Pipeline` (so the x-axis uses original feature units), then plot PDPs for the **four features with the largest absolute coefficient**. In 2–3 sentences, explain why the PDP curves for logistic regression are S-shaped and contrast this with the shape you would expect from a decision tree.

<details>
<summary>Solution</summary>

```python
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=10000, random_state=42),
)
pipe.fit(X_train, y_train)

top4 = sorted_idx[:4].tolist()   # four features with largest |coef|

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
PartialDependenceDisplay.from_estimator(
    pipe, X_train,
    features=top4,
    feature_names=bc.feature_names,
    ax=axes,
    grid_resolution=100,
)
fig.suptitle("Partial Dependence Plots — Top 4 Features (P(benign))")
fig.tight_layout()
plt.show()
```

Because logistic regression has a linear decision boundary, each feature's log-odds contribution is linear in xⱼ — a fixed amount per unit change, regardless of where on the axis you are. When that linear shift is passed through the sigmoid to produce a probability, the result is an **S-shaped (sigmoid) curve**: the probability changes slowly near 0 and 1 (where the sigmoid is flat) and fastest in the middle (where the sigmoid is steepest).

A decision tree PDP, by contrast, would show a **staircase** pattern: the predicted probability is constant within each leaf interval and jumps abruptly at each split threshold. The smooth sigmoid shape of logistic regression is a direct visual signature of its linear-in-log-odds structure.

</details>

---

## Step 13 — Shapley Values

### Background

**Shapley values** (from cooperative game theory) answer the question: *"how much did feature j contribute to pushing this particular prediction above or below the baseline?"*

For a logistic regression model the log-odds output is a linear function of the (standardised) features:

> f(**x**) = **w** · **x** + b

For any linear model, the Shapley value of feature j for sample **x** has an exact closed-form expression:

> φⱼ(**x**) = wⱼ · (xⱼ − E[Xⱼ])

where E[Xⱼ] is the mean of feature j over a background dataset (here, the training set). The **baseline prediction** is f(E[**X**]) — the model output at the mean of all features — and the sum of all Shapley values exactly reconstructs the model output:

> f(**x**) = f(E[**X**]) + Σⱼ φⱼ(**x**)

This additive decomposition holds exactly because the model is linear in the log-odds. After standardisation, E[Xⱼ] ≈ 0 for all features, so φⱼ ≈ wⱼ · xⱼ — the Shapley value is approximately the coefficient times the standardised feature value.

### Task

1. Compute Shapley values analytically for all test samples using the formula above. Verify that the reconstruction error (|f(**x**) − baseline − Σφⱼ|) is near machine precision.
2. Plot a **summary bar chart** showing the mean |Shapley value| per feature for the top 15 features.
3. Plot a **waterfall chart** for the first test sample showing how each of the top-10 features pushes the log-odds from the baseline to the final prediction.
4. In 2–3 sentences, explain how Shapley values differ from raw coefficients as a tool for understanding individual predictions.

<details>
<summary>Solution</summary>

```python
bg_mean    = X_train_sc.mean(axis=0)                  # ≈ 0 after scaling
shap_vals  = (X_test_sc - bg_mean) * lr.coef_[0]     # shape (n_test, n_features)
base_value = float(lr.decision_function(bg_mean.reshape(1, -1))[0])

log_odds  = lr.decision_function(X_test_sc)
residuals = np.abs(log_odds - (base_value + shap_vals.sum(axis=1)))
print(f"Shapley reconstruction error (max): {residuals.max():.2e}  (should be ~0)\n")
```

**Summary bar chart:**

```python
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
shap_order    = np.argsort(mean_abs_shap)[::-1]
top_shap_n    = 15

print("=== Shapley Values — Mean |φⱼ| across test set ===")
for rank, idx in enumerate(shap_order[:top_shap_n], 1):
    print(f"  {rank:2}. {bc.feature_names[idx]:<35} mean |φ| = {mean_abs_shap[idx]:.4f}")
print()

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(
    range(top_shap_n),
    mean_abs_shap[shap_order[:top_shap_n]][::-1],
)
ax.set_yticks(range(top_shap_n))
ax.set_yticklabels([bc.feature_names[i] for i in shap_order[:top_shap_n]][::-1])
ax.set_xlabel("Mean |Shapley value| (log-odds units)")
ax.set_title(f"Shapley Summary — Top {top_shap_n} Features")
fig.tight_layout()
plt.show()
```

**Waterfall chart for a single prediction:**

```python
sample_idx = 0
sv         = shap_vals[sample_idx]
order      = np.argsort(np.abs(sv))[::-1][:10]    # top-10 contributors
running    = base_value + np.cumsum(sv[order])
starts     = np.concatenate([[base_value], running[:-1]])
bar_colors = ["steelblue" if v > 0 else "tomato" for v in sv[order]]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(
    range(len(order)),
    sv[order][::-1],
    left=starts[::-1],
    color=bar_colors[::-1],
)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([bc.feature_names[i] for i in order][::-1])
ax.axvline(base_value, color="black", linewidth=0.8, linestyle="--",
           label=f"baseline = {base_value:.2f}")
ax.set_xlabel("Log-odds")
ax.set_title(f"Shapley Waterfall — test sample {sample_idx} "
             f"(true: {bc.target_names[y_test[sample_idx]]})")
ax.legend()
fig.tight_layout()
plt.show()
```

**Coefficients vs. Shapley values:** Raw coefficients describe the model's average sensitivity to a feature — how much the log-odds change per standard deviation, regardless of the actual value of that feature. Shapley values are **instance-specific**: they account for *how far* the feature value sits from the background mean, so a feature with a large coefficient but an unremarkable value for this particular sample will have a small Shapley value. This makes Shapley values the right tool for explaining a single prediction, while coefficients describe the model's global structure.

The waterfall chart makes this per-sample decomposition visual: each bar starts where the previous one ended, and the final position on the log-odds axis maps through the sigmoid to the model's output probability.

</details>
