"""
Gradient Boosting with scikit-learn — Step-by-Step Tutorial
=============================================================

## Introduction to Gradient Boosting

**Gradient boosting** is an ensemble learning algorithm that combines many
decision trees, like a random forest — but where a random forest builds all
its trees independently and averages them, gradient boosting builds trees
**sequentially**, and each new tree is trained to correct the errors of the
ensemble built so far. It belongs to the **boosting** family of methods, as
opposed to random forest's **bagging** family.

### How it works

Training proceeds as a loop:

1. Start with a constant prediction — typically the mean of the target.
2. Compute residuals — actual minus current prediction.
3. Fit a shallow tree to the residuals (not the original target).
4. Add a shrunk version of that tree's prediction to the ensemble:
   current_prediction += learning_rate * tree.predict(X)
5. Repeat for n_estimators rounds.

### Key hyperparameters

| Parameter          | Role                                                            |
|---------------------|------------------------------------------------------------------|
| `n_estimators`      | Number of boosting rounds — more can eventually overfit         |
| `learning_rate`     | Shrinkage per round — lower needs more trees but generalizes better |
| `max_depth`         | Per-tree depth cap — boosting trees stay shallow (2-4) on purpose |
| `subsample`         | Fraction of training data per tree (stochastic gradient boosting)|
| `n_iter_no_change`  | Early stopping — halt when validation score stops improving      |

### Random forest vs. gradient boosting

| Property               | Random Forest              | Gradient Boosting              |
|-------------------------|-----------------------------|----------------------------------|
| Tree construction       | Parallel, independent      | Sequential, each corrects last  |
| Individual trees        | Deep, low-bias, high-var   | Shallow, high-bias, low-var     |
| Reduces                 | Variance (averaging)       | Bias (sequential correction)    |
| Overfitting behavior    | More trees rarely hurts    | More trees CAN eventually overfit|
| Hyperparameter sensitivity | Fairly robust with defaults | Needs learning_rate/n_estimators tuned together |

---

## Preparation — Imports

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

- **numpy** — numerical operations and array handling
- **matplotlib** — plotting and visualisation
- **sklearn.datasets** — built-in datasets (Diabetes)
- **sklearn.model_selection** — splitting data into train/test sets
- **sklearn.tree** — DecisionTreeRegressor, used both to hand-build boosting
  in Step 3 and as the base learner sklearn uses internally
- **sklearn.ensemble** — GradientBoostingRegressor and RandomForestRegressor
- **sklearn.metrics** — tools to evaluate regression performance

---

## Step 1 — Load & Explore the Data

Load the Diabetes dataset. Assign the feature matrix to X and the target
vector to y. Print the number of samples, feature names, and the
range/mean/std of the target.

## Step 2 — Split into Train / Test Sets

Split the data 80/20 with a fixed random seed for reproducibility. No
stratify argument — the target is continuous, not a class label.

## Step 3 — Build Intuition: Fit Residuals by Hand

Implement three rounds of boosting manually: start from a constant
prediction (the training mean), and in a loop, fit a depth-2
DecisionTreeRegressor to the current residuals, scale its prediction by a
learning rate of 0.5, and add it to the running prediction. Print the
residual sum of squares (RSS) after each round.

## Step 4 — Train a Gradient Boosting Regressor

Train a GradientBoostingRegressor with 200 estimators, learning_rate=0.05,
and max_depth=3. Use random_state=42. Print the training R^2.

## Step 5 — Evaluate the Model

Generate predictions on the test set. Print MSE and R^2 for the gradient
boosting model, and train a RandomForestRegressor with matching
n_estimators/max_depth as a comparison baseline.

## Step 6 — Understand Feature Importance

Extract and print the feature importances, ranked from most to least
important.

## Step 7 — Visualise

  7a — Bar chart of RSS after each round of the from-scratch loop (Step 3).
  7b — Train vs. test MSE at every boosting round (staged_predict), marking
       the round with the lowest test MSE.
  7c — Horizontal bar chart of feature importances.

Do not save figures to disk — display them inline.

## Step 8 — Learning Rate vs. n_estimators Tradeoff

Train models across a grid of learning_rate x n_estimators. Record test R^2
for each combination and plot one curve per learning rate. In 2-3 sentences,
describe the tradeoff between learning_rate and n_estimators.

## Step 9 — Make a Single Prediction (Inference)

Using the trained gradient boosting model, predict the disease progression
score for the first sample in the test set. Print the predicted value
alongside the true label.

---
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# Step 1 — Load & Explore the Data
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# diabetes = load_diabetes()
#
# X = diabetes.data     # shape: (442, 10)
# y = diabetes.target   # disease progression score, one year after baseline
#
# print("=== Dataset Overview ===")
# print(f"Samples:  {X.shape[0]}")
# print(f"Features: {X.shape[1]}")
# print(f"Feature names: {diabetes.feature_names}")
# print(f"Target range: {y.min():.1f} to {y.max():.1f}  (mean={y.mean():.1f}, std={y.std():.1f})\n")
#
# The Diabetes dataset contains 442 patients described by 10 baseline
# measurements (age, sex, BMI, blood pressure, six serum measurements s1-s6).
# The target is a continuous disease-progression score, making this a
# regression problem — unlike the classification datasets in the decision
# tree / random forest tutorials.
# </details>

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Feature names: {diabetes.feature_names}")
print(f"Target range: {y.min():.1f} to {y.max():.1f}  (mean={y.mean():.1f}, std={y.std():.1f})\n")

# =============================================================================
# Step 2 — Split into Train / Test Sets
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
# )
#
# print(f"Training samples: {len(X_train)}")
# print(f"Test samples:     {len(X_test)}\n")
# </details>

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}\n")

# =============================================================================
# Step 3 — Build Intuition: Fit Residuals by Hand
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# learning_rate = 0.5
# n_rounds = 3
#
# current_pred_train = np.full(shape=y_train.shape, fill_value=y_train.mean())
# residuals = y_train - current_pred_train
# print(f"Round 0 (constant mean prediction): RSS = {np.sum(residuals**2):,.0f}")
#
# trees = []
# for round_num in range(1, n_rounds + 1):
#     tree = DecisionTreeRegressor(max_depth=2, random_state=42)
#     tree.fit(X_train, residuals)          # fit to the RESIDUALS, not y_train
#     trees.append(tree)
#
#     current_pred_train = current_pred_train + learning_rate * tree.predict(X_train)
#     residuals = y_train - current_pred_train
#
#     print(f"Round {round_num}: RSS = {np.sum(residuals**2):,.0f}")
#
# Each tree never sees the original target y_train — only the residuals left
# over from the current ensemble. As rounds progress, RSS shrinks because each
# new tree chips away at whatever error remains. GradientBoostingRegressor
# does exactly this, just with more rounds, a smaller learning_rate, and a
# more general loss-function gradient in place of the raw residual (residual
# and negative gradient coincide exactly for squared-error loss).
# </details>

print("=== From-Scratch Gradient Boosting (3 rounds) ===")

learning_rate_scratch = 0.5
n_rounds = 3

current_pred_train = np.full(shape=y_train.shape, fill_value=y_train.mean())
residuals = y_train - current_pred_train
print(f"Round 0 (constant mean prediction): RSS = {np.sum(residuals**2):,.0f}")

scratch_trees = []
rss_by_round = [float(np.sum(residuals**2))]
for round_num in range(1, n_rounds + 1):
    tree = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree.fit(X_train, residuals)
    scratch_trees.append(tree)

    current_pred_train = current_pred_train + learning_rate_scratch * tree.predict(X_train)
    residuals = y_train - current_pred_train

    rss = float(np.sum(residuals**2))
    rss_by_round.append(rss)
    print(f"Round {round_num}: RSS = {rss:,.0f}")
print()

# =============================================================================
# Step 4 — Train a Gradient Boosting Regressor
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# gbr = GradientBoostingRegressor(
#     n_estimators=200,
#     learning_rate=0.05,
#     max_depth=3,
#     random_state=42,
# )
# gbr.fit(X_train, y_train)
#
# train_r2 = r2_score(y_train, gbr.predict(X_train))
# print("=== Training ===")
# print(f"Training R^2: {train_r2:.4f}\n")
#
# n_estimators — 200 boosting rounds. Each adds a small correction; too many
#                can start fitting noise (see Step 7).
# learning_rate — 0.05 heavily discounts each tree's correction. Small values
#                 are the norm — trading training speed for a smoother fit.
# max_depth     — 3 keeps each tree "weak" on purpose. Deep trees here would
#                 let a single round overfit the residuals.
# </details>

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
gbr.fit(X_train, y_train)

train_r2 = r2_score(y_train, gbr.predict(X_train))
print("=== Training ===")
print(f"Training R^2: {train_r2:.4f}\n")

# =============================================================================
# Step 5 — Evaluate the Model
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# rf = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42)
# rf.fit(X_train, y_train)
#
# gbr_pred = gbr.predict(X_test)
# rf_pred = rf.predict(X_test)
#
# print("=== Evaluation ===")
# print(f"Gradient Boosting — MSE: {mean_squared_error(y_test, gbr_pred):.2f}  R^2: {r2_score(y_test, gbr_pred):.4f}")
# print(f"Random Forest      — MSE: {mean_squared_error(y_test, rf_pred):.2f}  R^2: {r2_score(y_test, rf_pred):.4f}\n")
#
# With un-tuned defaults, random forest edges out gradient boosting slightly
# here — on a small dataset (442 samples) with a fairly linear signal, RF's
# robustness to hyperparameters wins out. Gradient boosting's higher ceiling
# only shows up once learning_rate and n_estimators are tuned together
# (Step 8).
# </details>

rf = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

gbr_pred = gbr.predict(X_test)
rf_pred = rf.predict(X_test)

print("=== Evaluation ===")
print(f"Gradient Boosting — MSE: {mean_squared_error(y_test, gbr_pred):.2f}  R^2: {r2_score(y_test, gbr_pred):.4f}")
print(f"Random Forest      — MSE: {mean_squared_error(y_test, rf_pred):.2f}  R^2: {r2_score(y_test, rf_pred):.4f}\n")

# =============================================================================
# Step 6 — Understand Feature Importance
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# importances = gbr.feature_importances_
# sorted_idx = np.argsort(importances)[::-1]
#
# print("=== Feature Importances ===")
# for rank, idx in enumerate(sorted_idx, 1):
#     print(f"  {rank:2}. {diabetes.feature_names[idx]:<10} importance={importances[idx]:.4f}")
#
# bmi and s5 (a blood serum measurement) dominate — consistent with BMI being
# a well-known clinical predictor of diabetes progression.
# </details>

importances = gbr.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("=== Feature Importances ===")
for rank, idx in enumerate(sorted_idx, 1):
    print(f"  {rank:2}. {diabetes.feature_names[idx]:<10} importance={importances[idx]:.4f}")
print()

# =============================================================================
# Step 7 — Visualise
# =============================================================================
# <details>
# <summary>Solution</summary>

# --- 7a: Residual shrinking (from-scratch loop) ---
# Each bar is shorter than the last — direct evidence that fitting a tree to
# the residuals and adding it back (scaled by learning_rate) removes signal
# each round.

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(len(rss_by_round)), rss_by_round, tick_label=[f"Round {i}" for i in range(len(rss_by_round))])
ax.set_ylabel("Residual sum of squares")
ax.set_title("RSS Shrinking Across 3 Hand-Built Boosting Rounds")
fig.tight_layout()
plt.show()

# --- 7b: Staged train/test error ---
# Train MSE keeps falling for all 200 rounds. Test MSE bottoms out much
# earlier and then creeps back up: past that point, additional trees fit
# noise specific to the training set. This is what n_iter_no_change (early
# stopping) is designed to catch — random forest has no equivalent curve,
# since averaging independent trees can't overfit this way.

train_mse_stages = []
test_mse_stages = []
for train_pred_stage, test_pred_stage in zip(gbr.staged_predict(X_train), gbr.staged_predict(X_test)):
    train_mse_stages.append(mean_squared_error(y_train, train_pred_stage))
    test_mse_stages.append(mean_squared_error(y_test, test_pred_stage))

best_round = int(np.argmin(test_mse_stages)) + 1
print(f"Best test MSE at round {best_round} of {len(test_mse_stages)}: {min(test_mse_stages):.2f}\n")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(train_mse_stages) + 1), train_mse_stages, label="Train MSE")
ax.plot(range(1, len(test_mse_stages) + 1), test_mse_stages, label="Test MSE")
ax.axvline(best_round, color="gray", linestyle="--", label=f"Best test MSE (round {best_round})")
ax.set_xlabel("Boosting round (number of trees)")
ax.set_ylabel("MSE")
ax.set_title("Staged Train/Test Error — Gradient Boosting")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
plt.show()

# --- 7c: Feature importance bar chart ---

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(range(len(importances)), importances[sorted_idx][::-1])
ax.set_yticks(range(len(importances)))
ax.set_yticklabels([diabetes.feature_names[i] for i in sorted_idx][::-1])
ax.set_xlabel("Importance (mean decrease in impurity)")
ax.set_title("Feature Importances — Gradient Boosting")
fig.tight_layout()
plt.show()

# </details>

# =============================================================================
# Step 8 — Learning Rate vs. n_estimators Tradeoff
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# learning_rates = [0.01, 0.05, 0.1, 0.3]
# n_estimators_values = [10, 25, 50, 100, 200, 400]
#
# results = {}
# for lr in learning_rates:
#     scores = []
#     for n in n_estimators_values:
#         m = GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=3, random_state=42)
#         m.fit(X_train, y_train)
#         scores.append(r2_score(y_test, m.predict(X_test)))
#     results[lr] = scores
#
# A low learning rate (0.01) needs hundreds of rounds just to catch up — at
# n_estimators=10 it barely beats a constant prediction — but keeps improving
# all the way to 400 rounds, ending as the best score in the whole grid. A
# high learning rate (0.3) makes fast progress early but overfits and
# degrades with more rounds, since each oversized correction increasingly
# chases training noise. Rule of thumb: pick the smallest learning_rate you
# can afford, paired with enough n_estimators (or early stopping) to let it
# converge — trading training time for a better-generalizing model.
# </details>

learning_rates = [0.01, 0.05, 0.1, 0.3]
n_estimators_values = [10, 25, 50, 100, 200, 400]

print("=== Learning Rate vs n_estimators ===")
results = {}
for lr in learning_rates:
    scores = []
    for n in n_estimators_values:
        m = GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=3, random_state=42)
        m.fit(X_train, y_train)
        scores.append(r2_score(y_test, m.predict(X_test)))
    results[lr] = scores
    print(f"  lr={lr:<5} {[f'{s:.3f}' for s in scores]}")
print()

fig, ax = plt.subplots(figsize=(8, 5))
for lr, scores in results.items():
    ax.plot(n_estimators_values, scores, marker="o", label=f"learning_rate={lr}")
ax.set_xscale("log")
ax.set_xlabel("n_estimators (log scale)")
ax.set_ylabel("Test R^2")
ax.set_title("Learning Rate vs. n_estimators Tradeoff")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
plt.show()

# =============================================================================
# Step 9 — Make a Single Prediction (Inference)
# =============================================================================
# <details>
# <summary>Solution</summary>
#
# sample = X_test[[0]]
# prediction = gbr.predict(sample)[0]
# true_value = y_test[0]
#
# print("=== Single Prediction ===")
# print(f"Predicted disease progression score: {prediction:.1f}")
# print(f"True disease progression score:      {true_value:.1f}")
#
# Unlike the classification tutorials, predict() here returns a continuous
# value directly — there's no predict_proba step, since regression has no
# notion of class probability.
# </details>

sample = X_test[[0]]
prediction = gbr.predict(sample)[0]
true_value = y_test[0]

print("=== Single Prediction ===")
print("Input features (first test sample):")
for name, val in zip(diabetes.feature_names, sample[0]):
    print(f"  {name:<10} {val:.4f}")
print(f"\nPredicted disease progression score: {prediction:.1f}")
print(f"True disease progression score:      {true_value:.1f}")
