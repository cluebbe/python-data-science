"""
Logistic Regression with scikit-learn — Step-by-Step Tutorial
==============================================================

## Introduction to Logistic Regression

**Logistic regression** is a linear classification algorithm that models the
probability that a sample belongs to a class using the **sigmoid (logistic)
function**. Despite its name it is a *classifier*, not a regressor: it outputs
a probability between 0 and 1 and assigns the sample to the class with the
higher probability.

### How it works

For binary classification, logistic regression learns a weight vector **w** and
bias b that map the input features to a log-odds score, then squashes it through
the sigmoid:

    z = w · x + b              (linear combination of features)
    p = σ(z) = 1 / (1 + e⁻ᶻ)  (sigmoid → probability of class 1)
    ŷ = 1  if p ≥ 0.5,  else 0

Training minimises the **binary cross-entropy (log-loss)**:

    L = −(1/n) Σ [ yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ) ]

The decision boundary is a *hyperplane* in feature space (a line in 2-D).
Logistic regression captures global linear separability but cannot model
curved or axis-aligned boundaries the way tree-based methods can.

### Regularisation

Without regularisation the model can overfit, especially with many or correlated
features. scikit-learn adds an L2 (ridge) penalty by default:

    L_reg = L + (1 / 2C) Σ wⱼ²

The hyperparameter **C** is the *inverse* of regularisation strength:
  - Small C  → strong regularisation → weights shrink toward zero  → underfitting risk
  - Large C  → weak regularisation  → weights can grow freely      → overfitting risk

### Key hyperparameters

| Parameter   | Role                                                           |
|-------------|----------------------------------------------------------------|
| `C`         | Inverse regularisation strength — primary tuning lever         |
| `penalty`   | `"l2"` (default, ridge) or `"l1"` (lasso, produces sparsity)  |
| `solver`    | Optimisation algorithm — `"lbfgs"` for small/medium datasets  |
| `max_iter`  | Maximum solver iterations — raise if ConvergenceWarning fires  |

### Feature scaling is essential

Tree-based methods are invariant to feature scale — they compare relative values
within one feature at a time. Logistic regression is not: features with large
magnitude dominate the dot product w · x, making training unstable and
coefficients incomparable. Always **standardise** features (zero mean, unit
variance) before fitting a logistic regression.

### Logistic regression vs. tree-based classifiers

| Property              | Logistic Regression       | Decision Tree / RF              |
|-----------------------|---------------------------|---------------------------------|
| Decision boundary     | Linear (hyperplane)       | Axis-aligned splits (piecewise) |
| Feature scaling       | Required                  | Not needed                      |
| Probability output    | Well-calibrated           | Often poorly calibrated         |
| Interpretability      | Signed coefficients       | Feature importances / tree vis  |
| High-dim sparse data  | Works well                | Slower, prone to overfitting    |
| Non-linear patterns   | Misses them               | Captures them naturally         |

---

## Preparation — Imports

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        ConfusionMatrixDisplay,
        roc_auc_score,
        RocCurveDisplay,
    )

- **numpy** — numerical operations and array handling
- **matplotlib** — plotting and visualisation
- **StandardScaler** — zero-mean, unit-variance feature normalisation
- **LogisticRegression** — scikit-learn's logistic regression classifier
- **DecisionTreeClassifier** — used in Step 9 for comparison
- **roc_auc_score / RocCurveDisplay** — ROC curve and AUC metric

---

## Step 1 — Load & Explore the Data

Load the Breast Cancer Wisconsin dataset. Assign the feature matrix to `X` and
the target vector to `y`. Print the number of samples, feature names, class
names, and class distribution.

## Step 2 — Split into Train / Test Sets

Split the data 80/20. Ensure the split is reproducible and class proportions are
preserved in both sets.

## Step 3 — Scale the Features

Fit a `StandardScaler` on the **training data only** and apply it to both train
and test sets. Print the mean and standard deviation of the first feature before
and after scaling to verify the transformation.

## Step 4 — Train a Logistic Regression Classifier

Fit a `LogisticRegression` with `C=1.0` and `max_iter=10000`. Print the training
accuracy and the number of iterations the solver actually used.

## Step 5 — Evaluate the Model

Generate predictions on the test set. Print the test accuracy, the ROC-AUC score,
and a full classification report. Also print the predicted probability for the
first five test samples.

## Step 6 — Understand the Coefficients

Extract the model's coefficient vector and display all features sorted by absolute
coefficient magnitude. Explain what a large positive or negative coefficient means
for the predicted probability.

## Step 7 — Visualise

Display three plots:
  7a — Confusion matrix on the test set.
  7b — ROC curve with the AUC score annotated.
  7c — Coefficient bar chart for the top-15 most influential features, coloured
       by direction (positive = blue, negative = red).

Do not save figures to disk — display them inline.

## Step 8 — Regularisation: C vs. Accuracy

Train logistic regression models with C ∈ {0.001, 0.01, 0.1, 1, 10, 100}. Record
train and test accuracy for each value. Plot both curves on a log-scale x-axis.
In 2–3 sentences, explain when to increase and when to decrease C.

## Step 9 — Logistic Regression vs. Decision Tree

Train a `DecisionTreeClassifier` (max_depth=3, random_state=42) on the *unscaled*
training data (trees do not need scaling). Compare test accuracy and ROC-AUC
against the logistic regression model. In 2–3 sentences explain the result.

## Step 10 — Make a Single Prediction (Inference)

Using the trained logistic regression, predict the class and class probabilities
for the first sample in the dataset. Remember to scale the sample first. Print
the predicted label, true label, and the per-class probability distribution.
Explain what the probability output means in a clinical context.

---
"""

# =============================================================================
# Imports
# =============================================================================
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

# =============================================================================
# Step 1 — Load & Explore the Data
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# bc = load_breast_cancer()
#
# X = bc.data    # shape: (569, 30) — cell nucleus measurements
# y = bc.target  # 0=malignant, 1=benign
#
# print("=== Dataset Overview ===")
# print(f"Samples:  {X.shape[0]}")
# print(f"Features: {X.shape[1]}")
# print(f"Feature names: {bc.feature_names.tolist()}")
# print(f"Classes:  {bc.target_names}")
# print(f"Class distribution: {np.bincount(y)}  (0=malignant, 1=benign)\n")
#
# The Breast Cancer Wisconsin dataset contains 569 patient samples described by
# 30 numerical measurements of cell nuclei (radius, texture, perimeter, area,
# smoothness, etc.), each computed as the mean, standard error, and worst value
# across cells in the biopsy image. The binary target — malignant (212 samples)
# vs. benign (357 samples) — maps perfectly to a logistic regression problem:
# we want a calibrated probability of malignancy so borderline cases can be
# flagged for further review rather than receiving a hard yes/no decision.
# </details>

bc = load_breast_cancer()

X = bc.data
y = bc.target  # 0=malignant, 1=benign — sklearn's encoding is inverted from clinical convention
               # (clinically, malignant is "positive", but here it is encoded as 0)

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Feature names: {bc.feature_names.tolist()}")
print(f"Classes:  {bc.target_names}")
print(f"Class distribution: {np.bincount(y)}  (0=malignant, 1=benign)\n")

# =============================================================================
# Step 2 — Split into Train / Test Sets
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,    # 20% held out for evaluation
#     random_state=42,  # reproducibility
#     stratify=y,       # preserve the ~37/63 malignant/benign ratio in both sets
# )
#
# print(f"Training samples: {len(X_train)}")
# print(f"Test samples:     {len(X_test)}\n")
#
# - test_size=0.2  → ~114 test / ~455 train
# - stratify=y     → class proportions match in both splits, so the test set
#                    is representative and accuracy is not skewed by imbalance
# </details>

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}\n")

# =============================================================================
# Step 3 — Scale the Features
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# scaler = StandardScaler()
# X_train_sc = scaler.fit_transform(X_train)  # fit on train, then transform
# X_test_sc  = scaler.transform(X_test)       # transform only — never fit on test
#
# print("=== Feature Scaling ===")
# print(f"Before scaling — mean: {X_train[:, 0].mean():.3f}  std: {X_train[:, 0].std():.3f}")
# print(f"After  scaling — mean: {X_train_sc[:, 0].mean():.3f}  std: {X_train_sc[:, 0].std():.3f}\n")
#
# Calling fit_transform only on the training set prevents data leakage: if we
# included the test set when computing the mean and std, the model would
# indirectly observe test distribution statistics during training, producing
# an overoptimistic performance estimate. The test set must be treated as
# entirely unseen — transform it with the scaler fitted on train data only.
# </details>

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("=== Feature Scaling ===")
print(f"Before scaling — mean: {X_train[:, 0].mean():.3f}  std: {X_train[:, 0].std():.3f}")
print(f"After  scaling — mean: {X_train_sc[:, 0].mean():.3f}  std: {X_train_sc[:, 0].std():.3f}\n")

# =============================================================================
# Step 4 — Train a Logistic Regression Classifier
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# lr = LogisticRegression(C=1.0, max_iter=10000, random_state=42)
# lr.fit(X_train_sc, y_train)
#
# train_acc = accuracy_score(y_train, lr.predict(X_train_sc))
# print("=== Training ===")
# print(f"Training accuracy: {train_acc:.2%}")
# print(f"Solver iterations: {lr.n_iter_[0]}\n")
#
# Key hyperparameters:
#
#   C          — Inverse regularisation strength. C=1 is the scikit-learn default
#                and a reasonable starting point; tune via cross-validation.
#
#   max_iter   — The solver (lbfgs by default) is iterative. On high-dimensional
#                data the default of 100 often triggers a ConvergenceWarning;
#                10 000 is generous. Check lr.n_iter_ to see how many were used —
#                if it equals max_iter the model did not converge and you should
#                raise the limit or rescale features more aggressively.
# </details>

lr = LogisticRegression(C=1.0, max_iter=10000, random_state=42)
lr.fit(X_train_sc, y_train)

train_acc = accuracy_score(y_train, lr.predict(X_train_sc))
print("=== Training ===")
print(f"Training accuracy: {train_acc:.2%}")
print(f"Solver iterations: {lr.n_iter_[0]}\n")

# =============================================================================
# Step 5 — Evaluate the Model
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# y_pred = lr.predict(X_test_sc)
# y_prob = lr.predict_proba(X_test_sc)[:, 1]  # P(benign) for each test sample
#
# test_acc = accuracy_score(y_test, y_pred)
# roc_auc  = roc_auc_score(y_test, y_prob)
#
# print("=== Evaluation ===")
# print(f"Test accuracy: {test_acc:.2%}")
# print(f"ROC-AUC:       {roc_auc:.4f}\n")
# print("Classification report:")
# print(classification_report(y_test, y_pred, target_names=bc.target_names))
#
# print("Predicted probabilities (first 5 test samples):")
# for i in range(5):
#     print(f"  sample {i}: p(benign)={y_prob[i]:.3f}  "
#           f"predicted={bc.target_names[y_pred[i]]}  "
#           f"true={bc.target_names[y_test[i]]}")
# print()
#
# ROC-AUC measures the probability that the model ranks a randomly chosen benign
# sample above a randomly chosen malignant one, across all decision thresholds.
# An AUC near 1.0 means the model separates the classes almost perfectly, even
# before you choose a threshold. AUC is preferable to accuracy for imbalanced
# datasets because it is unaffected by the class ratio — a classifier that always
# predicts "benign" achieves 63% accuracy but an AUC of exactly 0.5.
# </details>

y_pred = lr.predict(X_test_sc)
y_prob = lr.predict_proba(X_test_sc)[:, 1]

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

# =============================================================================
# Step 6 — Understand the Coefficients
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# coefs = lr.coef_[0]          # shape (30,) — one coefficient per feature
# sorted_idx = np.argsort(np.abs(coefs))[::-1]
#
# print("=== Model Coefficients (sorted by |coef|) ===")
# for rank, idx in enumerate(sorted_idx, 1):
#     direction = "→ benign" if coefs[idx] > 0 else "→ malignant"
#     print(f"  {rank:2}. {bc.feature_names[idx]:<35} coef={coefs[idx]:+.4f}  {direction}")
# print()
#
# Because features are standardised, coefficients are directly comparable in
# magnitude: a coefficient of +2 means a one-standard-deviation increase in that
# feature raises the log-odds of being benign by 2 — roughly tripling the benign
# odds. A large negative coefficient points toward malignancy. Features with
# coefficients near zero have little marginal influence on the prediction.
# Unlike a random forest's feature importances, coefficients carry sign: they
# tell you not just which features matter but in which direction they push
# the prediction.
# </details>

coefs = lr.coef_[0]
sorted_idx = np.argsort(np.abs(coefs))[::-1]

print("=== Model Coefficients (sorted by |coef|) ===")
for rank, idx in enumerate(sorted_idx, 1):
    direction = "→ benign" if coefs[idx] > 0 else "→ malignant"
    print(f"  {rank:2}. {bc.feature_names[idx]:<35} coef={coefs[idx]:+.4f}  {direction}")
print()

# =============================================================================
# Step 7 — Visualise
# =============================================================================
# <details>
# <summary>Solution</summary>

# --- 7a: Confusion matrix ---
# For cancer classification the most dangerous cell is the top-right:
# false negatives (malignant predicted as benign). Missing a malignant tumour
# is far more costly than a false alarm, so consider lowering the threshold
# below 0.5 to improve malignant recall at the expense of benign precision.

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

# --- 7b: ROC curve ---
# The ROC curve plots True Positive Rate vs False Positive Rate across all
# classification thresholds. A perfect classifier hugs the top-left corner;
# a random classifier follows the diagonal. The area under the curve (AUC)
# summarises discrimination ability in a single number: 0.5 = random, 1.0 = perfect.

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(lr, X_test_sc, y_test, ax=ax, name="Logistic Regression")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
ax.set_title(f"ROC Curve — Logistic Regression (AUC = {roc_auc:.3f})")
ax.legend()
fig.tight_layout()
plt.show()

# --- 7c: Coefficient bar chart (top 15 by magnitude) ---
# Blue bars → feature increases P(benign); red bars → increases P(malignant).
# Because features are standardised, bar lengths are directly comparable —
# longer bars drive the prediction more forcefully.

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

# </details>

# =============================================================================
# Step 8 — Regularisation: C vs. Accuracy
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# C_values = [0.001, 0.01, 0.1, 1, 10, 100]
# train_scores, test_scores = [], []
#
# print("=== Regularisation Strength (C) vs Accuracy ===")
# for C in C_values:
#     m = LogisticRegression(C=C, max_iter=10000, random_state=42)
#     m.fit(X_train_sc, y_train)
#     tr = accuracy_score(y_train, m.predict(X_train_sc))
#     te = accuracy_score(y_test,  m.predict(X_test_sc))
#     train_scores.append(tr)
#     test_scores.append(te)
#     print(f"  C={C:<8}  train={tr:.2%}  test={te:.2%}")
#
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.plot(C_values, train_scores, marker="o", label="Train accuracy")
# ax.plot(C_values, test_scores,  marker="s", label="Test accuracy")
# ax.set_xscale("log")
# ax.set_xlabel("C (log scale)")
# ax.set_ylabel("Accuracy")
# ax.set_title("Regularisation Strength vs Accuracy — Logistic Regression")
# ax.legend()
# ax.grid(True, linestyle="--", alpha=0.5)
# fig.tight_layout()
# plt.show()
#
# At very small C (strong regularisation) both train and test accuracy are
# depressed — the weights are shrunk so aggressively that the model cannot fit
# even the training data (underfitting). As C increases the model gains
# flexibility and accuracy rises for both sets. If you see train accuracy
# continue to climb while test accuracy plateaus or falls, you are crossing into
# overfitting territory — that is the signal to reduce C (strengthen regularisation).
# For this dataset C=1 sits near the sweet spot; the Breast Cancer features are
# well-scaled and mostly linearly separable, so heavy regularisation is not needed.
# </details>

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

# =============================================================================
# Step 9 — Logistic Regression vs. Decision Tree
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# dt = DecisionTreeClassifier(max_depth=3, random_state=42)
# dt.fit(X_train, y_train)  # unscaled — trees are scale-invariant
# dt_pred = dt.predict(X_test)
#
# dt_acc = accuracy_score(y_test, dt_pred)
# lr_acc = accuracy_score(y_test, y_pred)
# dt_auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
#
# print("=== Logistic Regression vs. Decision Tree ===")
# print(f"  Logistic Regression — accuracy: {lr_acc:.2%}  AUC: {roc_auc:.4f}")
# print(f"  Decision Tree       — accuracy: {dt_acc:.2%}  AUC: {dt_auc:.4f}")
# print(f"  Accuracy gap: {lr_acc - dt_acc:+.2%}\n")
#
# On this dataset the classes are largely linearly separable — logistic regression
# finds a single hyperplane that cleanly separates most malignant from benign
# samples, while a depth-3 tree can only approximate that boundary with three
# axis-aligned cuts. The AUC gap is often more revealing than accuracy: a higher
# AUC means the logistic model's probability scores rank samples more reliably
# across all thresholds. For problems with strongly non-linear class boundaries
# a deeper tree (or a random forest) would close or reverse this gap.
# </details>

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)
lr_acc = accuracy_score(y_test, y_pred)
dt_auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])

print("=== Logistic Regression vs. Decision Tree ===")
print(f"  Logistic Regression — accuracy: {lr_acc:.2%}  AUC: {roc_auc:.4f}")
print(f"  Decision Tree       — accuracy: {dt_acc:.2%}  AUC: {dt_auc:.4f}")
print(f"  Accuracy gap: {lr_acc - dt_acc:+.2%}\n")

# =============================================================================
# Step 10 — Make a Single Prediction (Inference)
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# sample_raw = bc.data[[0]]                  # first sample — known malignant (label=0)
# sample_sc  = scaler.transform(sample_raw)  # scale with the already-fitted scaler
# prediction    = lr.predict(sample_sc)[0]
# probabilities = lr.predict_proba(sample_sc)[0]
#
# print("=== Single Prediction ===")
# print("Input features (first sample from dataset):")
# for name, val in zip(bc.feature_names, sample_raw[0]):
#     print(f"  {name:<35} {val:.4f}")
# print(f"\nPredicted class: {bc.target_names[prediction]}")
# print(f"True label:      {bc.target_names[bc.target[0]]}")
# print("Class probabilities:")
# for cls, prob in zip(bc.target_names, probabilities):
#     print(f"  {cls:<12} {prob:.2%}")
#
# Unlike a decision tree — which assigns a hard probability based on the
# fraction of training samples in a leaf — logistic regression's sigmoid output
# is well-calibrated: a predicted P(malignant) = 0.97 genuinely reflects ~97%
# confidence. In clinical decision support, calibrated probabilities allow
# clinicians to set an evidence-based threshold (e.g. flag any sample with
# P(malignant) > 0.30 for biopsy) rather than relying on the fixed 0.5 cutoff.
# </details>

sample_raw = bc.data[[0]]
sample_sc  = scaler.transform(sample_raw)
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

# =============================================================================
# Step 11 — Partial Dependence Plots (PDPs)
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# A Partial Dependence Plot (PDP) shows the *marginal* effect of one feature on
# the predicted probability, averaged over the joint distribution of all other
# features. For each grid value of feature j, PDP(xⱼ) is:
#
#     PDP(xⱼ) = (1/n) Σᵢ f(x₁ⁱ, …, xⱼ, …, xₙⁱ)
#
# The model computes a prediction for every training sample with xⱼ fixed at
# the grid value; the average over samples removes the effect of all other
# features. The resulting curve tells you: "if I could dial feature j up or
# down while everything else stayed realistic, how would the predicted
# probability change?"
#
# We wrap the scaler and classifier in a Pipeline so the PDP can operate on
# original-scale features — the x-axis is then interpretable in raw units.
#
# pipe = make_pipeline(
#     StandardScaler(),
#     LogisticRegression(C=1.0, max_iter=10000, random_state=42),
# )
# pipe.fit(X_train, y_train)
#
# top4 = sorted_idx[:4].tolist()   # four most influential features by |coef|
#
# fig, axes = plt.subplots(1, 4, figsize=(16, 4))
# PartialDependenceDisplay.from_estimator(
#     pipe, X_train,
#     features=top4,
#     feature_names=bc.feature_names,
#     ax=axes,
#     grid_resolution=100,
# )
# fig.suptitle("Partial Dependence Plots — Top 4 Features (P(benign))")
# fig.tight_layout()
# plt.show()
#
# Because logistic regression has a linear decision boundary, the PDPs are
# S-shaped (sigmoid) curves — a one-unit change in a feature shifts the
# log-odds by a fixed amount regardless of where on the axis you are, but that
# shift is translated to probability space by the sigmoid, which compresses
# the ends of the [0,1] interval. Features with large negative coefficients
# (e.g. worst_texture) show a downward slope: as the raw measurement grows,
# P(benign) falls. PDPs for tree-based models would show staircase shapes
# instead of smooth sigmoids.
# </details>

pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=1.0, max_iter=10000, random_state=42),
)
pipe.fit(X_train, y_train)

top4 = sorted_idx[:4].tolist()
best_positive = 15  # compactness error — strongest positive coefficient (+0.68), slopes upward
pdp_features = top4 + [best_positive]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
PartialDependenceDisplay.from_estimator(
    pipe, X_train,
    features=pdp_features,
    feature_names=bc.feature_names,
    ax=axes,
    grid_resolution=100,
)
fig.suptitle("Partial Dependence Plots — Top 4 Features + compactness error (P(benign))")
fig.tight_layout()
plt.show()

# =============================================================================
# Step 12 — Shapley Values
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# Shapley values (from cooperative game theory) answer: "how much did feature j
# contribute to pushing this prediction above or below the baseline?"
#
# For a logistic regression model, the log-odds output is a linear function:
#
#     f(x) = w · x + b
#
# For linear models, the Shapley value of feature j for sample x is exact and
# analytically simple:
#
#     φⱼ(x) = wⱼ · (xⱼ − E[Xⱼ])
#
# where E[Xⱼ] is the mean of feature j over the background (training) dataset.
# The baseline prediction is f(E[X]) = w · E[X] + b, and the sum of all
# Shapley values exactly reconstructs the model output:
#
#     f(x) = f(E[X]) + Σⱼ φⱼ(x)
#
# Because we standardised the features, E[Xⱼ] ≈ 0 for all j, so
# φⱼ ≈ wⱼ · xⱼ — the Shapley value is the coefficient times the (standardised)
# feature value.
#
# bg_mean    = X_train_sc.mean(axis=0)                 # ≈ 0 after scaling
# shap_vals  = (X_test_sc - bg_mean) * lr.coef_[0]    # shape (n_test, n_features)
# base_value = float(lr.decision_function(bg_mean.reshape(1, -1))[0])
#
# # Sanity check: base + sum(shap) == model log-odds output
# log_odds   = lr.decision_function(X_test_sc)
# residuals  = np.abs(log_odds - (base_value + shap_vals.sum(axis=1)))
# print(f"Max reconstruction error: {residuals.max():.2e}  (should be ~0)\n")
#
# # --- Waterfall plot for a single prediction ---
# # A waterfall chart shows how each feature pushes the log-odds above or below
# # the baseline for one specific sample.
#
# sample_idx  = 1   # malignant sample — clean cascade
# sv          = shap_vals[sample_idx]
# order       = np.argsort(np.abs(sv))[::-1][:10]   # top-10, largest |sv| first
# running     = base_value + np.cumsum(sv[order])
# starts      = np.concatenate([[base_value], running[:-1]])
# bar_colors  = ["steelblue" if v > 0 else "tomato" for v in sv[order]]
#
# y_pos = list(range(len(order) - 1, -1, -1))  # largest at top
#
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.barh(y_pos, sv[order], left=starts, color=bar_colors)
# ax.set_yticks(y_pos)
# ax.set_yticklabels(bc.feature_names[order])
# ax.axvline(base_value, color="black", linewidth=0.8, linestyle="--",
#            label=f"baseline = {base_value:.2f}")
# ax.set_xlabel("Log-odds")
# ax.set_title(f"Shapley Waterfall — test sample {sample_idx} "
#              f"(true: {bc.target_names[y_test[sample_idx]]})")
# ax.legend()
# fig.tight_layout()
# plt.show()
#
# The waterfall chart explains a single prediction. Each bar
# starts where the previous one ended — blue bars push the log-odds higher
# (toward benign), red bars push it lower (toward malignant). The final
# log-odds value maps through the sigmoid to the model's output probability.
# </details>

bg_mean   = X_train_sc.mean(axis=0)
shap_vals = (X_test_sc - bg_mean) * lr.coef_[0]
base_value = float(lr.decision_function(bg_mean.reshape(1, -1))[0])

log_odds  = lr.decision_function(X_test_sc)
residuals = np.abs(log_odds - (base_value + shap_vals.sum(axis=1)))
print(f"Shapley reconstruction error (max): {residuals.max():.2e}  (should be ~0)\n")

sample_idx = 1  # malignant sample — features push mostly in one direction for a clean cascade
sv    = shap_vals[sample_idx]
order = np.argsort(np.abs(sv))[::-1][:10]   # largest |sv| first

running    = base_value + np.cumsum(sv[order])
starts     = np.concatenate([[base_value], running[:-1]])
bar_colors = ["steelblue" if v > 0 else "tomato" for v in sv[order]]

# largest feature at top (y=9), smallest at bottom (y=0)
y_pos = list(range(len(order) - 1, -1, -1))

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(y_pos, sv[order], left=starts, color=bar_colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(bc.feature_names[order])
ax.axvline(base_value, color="black", linewidth=0.8, linestyle="--",
           label=f"baseline = {base_value:.2f}")
ax.set_xlabel("Log-odds")
ax.set_title(f"Shapley Waterfall — test sample {sample_idx} "
             f"(true: {bc.target_names[y_test[sample_idx]]})")
ax.legend()
fig.tight_layout()
plt.show()
