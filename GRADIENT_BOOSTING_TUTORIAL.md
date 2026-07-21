# Gradient Boosting with scikit-learn — Step-by-Step Tutorial

## Introduction to Gradient Boosting

**Gradient boosting** is an ensemble learning algorithm that combines many decision trees, like a random forest — but where a random forest builds all its trees independently and averages them, gradient boosting builds trees **sequentially**, and each new tree is trained to correct the errors of the ensemble built so far. It belongs to the **boosting** family of methods, as opposed to random forest's **bagging** family.

### How it works

Training proceeds as a loop:

1. **Start with a constant prediction** — typically the mean of the target (the best guess before seeing any features).
2. **Compute residuals** — for every training sample, `residual = actual - current_prediction`. This is what the ensemble is still getting wrong.
3. **Fit a shallow tree to the residuals** — not to the original target. The tree learns which feature combinations are associated with the largest remaining errors.
4. **Add a shrunk version of that tree's prediction to the ensemble** — `current_prediction += learning_rate * tree.predict(X)`. The `learning_rate` (shrinkage) controls how much each individual tree is allowed to influence the total.
5. **Repeat** for `n_estimators` rounds. Each round's tree only has to explain what's left over — a much easier problem than the original one.

### Why this is different from bagging

A single decision tree has high variance but a boosted ensemble's individual trees are shallow (often depth 2–4) and therefore individually high-**bias**, low-variance. Boosting reduces that bias by chaining many weak learners together, each specializing in the previous ensemble's mistakes. Random forest instead starts with low-bias, high-variance trees (deep, unconstrained) and reduces variance by averaging many of them. Same goal — a strong low-error model — opposite starting point.

### Key hyperparameters

| Parameter | Role |
|---|---|
| `n_estimators` | Number of boosting rounds (trees) — more rounds keep reducing training error, but can eventually overfit |
| `learning_rate` | Shrinkage — how much each tree's correction is trusted. Lower values need more trees but generalize better |
| `max_depth` | Per-tree depth cap — boosting trees are usually shallow (2–4), unlike random forest's deeper trees |
| `subsample` | Fraction of training data used per tree (stochastic gradient boosting) — adds randomness to reduce overfitting |
| `n_iter_no_change` | Early stopping — halt training if validation score stops improving for this many rounds |

### Random forest vs. gradient boosting

| Property | Random Forest | Gradient Boosting |
|---|---|---|
| Tree construction | Parallel, independent | Sequential, each corrects the last |
| Individual trees | Deep, low-bias, high-variance | Shallow, high-bias, low-variance |
| Reduces | Variance (via averaging) | Bias (via sequential correction) |
| Overfitting behavior | More trees rarely hurts | More trees *can* eventually overfit |
| Sensitivity to hyperparameters | Fairly robust with defaults | Needs `learning_rate`/`n_estimators` tuned together |
| Typical accuracy (tabular data) | Strong out-of-the-box | Often higher ceiling, with tuning |

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
mkdir gradient-boosting && cd gradient-boosting

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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

- **numpy** — numerical operations and array handling
- **matplotlib** — plotting and visualisation
- **sklearn.datasets** — built-in datasets (Diabetes)
- **sklearn.model_selection** — splitting data into train/test sets
- **sklearn.tree** — `DecisionTreeRegressor`, used both to hand-build boosting in Step 3 and as the base learner sklearn uses internally
- **sklearn.ensemble** — `GradientBoostingRegressor` and `RandomForestRegressor` (for comparison)
- **sklearn.metrics** — tools to evaluate regression performance

---

## Step 1 — Load & Explore the Data

Load the Diabetes dataset. Assign the feature matrix to `X` and the target vector to `y`. Print the number of samples, feature names, and the range/mean/std of the target.

<details>
<summary>Solution</summary>

```python
diabetes = load_diabetes()

X = diabetes.data     # shape: (442, 10)
y = diabetes.target   # quantitative measure of disease progression one year after baseline

print("=== Dataset Overview ===")
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Feature names: {diabetes.feature_names}")
print(f"Target range: {y.min():.1f} to {y.max():.1f}  (mean={y.mean():.1f}, std={y.std():.1f})\n")
```

Output:

```
=== Dataset Overview ===
Samples:  442
Features: 10
Feature names: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
Target range: 25.0 to 346.0  (mean=152.1, std=77.0)
```

The **Diabetes** dataset contains 442 patients, each described by 10 baseline measurements: age, sex, BMI, average blood pressure, and six blood serum measurements (`s1`–`s6`). The target is a continuous score of disease progression one year after baseline — this makes it a **regression** problem, unlike the classification datasets in the decision tree / random forest tutorials.

</details>

---

## Step 2 — Split into Train / Test Sets

Split the data into training and test sets using an 80/20 ratio, with a fixed random seed for reproducibility.

<details>
<summary>Solution</summary>

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% held out for evaluation
    random_state=42,  # reproducibility
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}\n")
```

Output:

```
Training samples: 353
Test samples:     89
```

Unlike the classification tutorials, there's no `stratify` argument here — stratification balances class labels, which doesn't apply to a continuous target.

</details>

---

## Step 3 — Build Intuition: Fit Residuals by Hand

Before reaching for `GradientBoostingRegressor`, implement three rounds of boosting manually: start from a constant prediction (the training mean), and in a loop, fit a depth-2 `DecisionTreeRegressor` to the current residuals, scale its prediction by a learning rate of `0.5`, and add it to the running prediction. Print the residual sum of squares (RSS) after each round.

<details>
<summary>Solution</summary>

```python
print("=== From-Scratch Gradient Boosting (3 rounds) ===")

learning_rate = 0.5
n_rounds = 3

current_pred_train = np.full(shape=y_train.shape, fill_value=y_train.mean())
residuals = y_train - current_pred_train
print(f"Round 0 (constant mean prediction): RSS = {np.sum(residuals**2):,.0f}")

trees = []
for round_num in range(1, n_rounds + 1):
    tree = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree.fit(X_train, residuals)          # fit to the RESIDUALS, not y_train
    trees.append(tree)

    current_pred_train = current_pred_train + learning_rate * tree.predict(X_train)
    residuals = y_train - current_pred_train

    print(f"Round {round_num}: RSS = {np.sum(residuals**2):,.0f}")
print()
```

Output:

```
=== From-Scratch Gradient Boosting (3 rounds) ===
Round 0 (constant mean prediction): RSS = 2,144,968
Round 1: RSS = 1,425,438
Round 2: RSS = 1,166,531
Round 3: RSS = 1,030,110
```

This is the entire algorithm in miniature: each tree never sees the original target `y_train` — only the residuals left over from the current ensemble. As rounds progress, RSS shrinks because each new tree chips away at whatever error remains. `GradientBoostingRegressor` in the next step does exactly this, just with more rounds, smaller `learning_rate`, and a more principled loss-function gradient in place of the raw residual (which is the special case for squared-error loss — "residual" and "negative gradient" coincide exactly when the loss is mean squared error).

</details>

---

## Step 4 — Train a Gradient Boosting Regressor

Train a `GradientBoostingRegressor` with 200 estimators, a learning rate of `0.05`, and max depth 3. Use `random_state=42`. Print the training R².

<details>
<summary>Solution</summary>

```python
gbr = GradientBoostingRegressor(
    n_estimators=200,   # boosting rounds
    learning_rate=0.05, # shrinkage per round
    max_depth=3,        # shallow trees — boosting trees stay weak on purpose
    random_state=42,
)
gbr.fit(X_train, y_train)

train_r2 = r2_score(y_train, gbr.predict(X_train))
print("=== Training ===")
print(f"Training R^2: {train_r2:.4f}\n")
```

Output:

```
=== Training ===
Training R^2: 0.8297
```

Key hyperparameters:

| Parameter | Effect |
|---|---|
| `n_estimators` | 200 boosting rounds. Each adds a small correction; too many can start fitting noise (see Step 7). |
| `learning_rate` | 0.05 means each tree's correction is heavily discounted. Small values are the norm in boosting — they trade training speed for a smoother, more generalizable fit. |
| `max_depth` | 3 keeps each tree "weak" on purpose. Deep trees here would let a single round overfit the residuals, defeating the point of correcting gradually. |

</details>

---

## Step 5 — Evaluate the Model

Generate predictions on the test set. Print MSE and R² for the gradient boosting model, and train a `RandomForestRegressor` with matching `n_estimators`/`max_depth` as a comparison baseline.

<details>
<summary>Solution</summary>

```python
rf = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

gbr_pred = gbr.predict(X_test)
rf_pred = rf.predict(X_test)

print("=== Evaluation ===")
print(f"Gradient Boosting — MSE: {mean_squared_error(y_test, gbr_pred):.2f}  R^2: {r2_score(y_test, gbr_pred):.4f}")
print(f"Random Forest      — MSE: {mean_squared_error(y_test, rf_pred):.2f}  R^2: {r2_score(y_test, rf_pred):.4f}\n")
```

Output:

```
=== Evaluation ===
Gradient Boosting — MSE: 2833.00  R^2: 0.4653
Random Forest      — MSE: 2774.29  R^2: 0.4764
```

With un-tuned defaults, random forest edges out gradient boosting slightly here — on a small dataset (442 samples) with a fairly linear signal, RF's robustness to hyperparameters wins out. This is a useful reality check: gradient boosting's higher ceiling only shows up once `learning_rate` and `n_estimators` are tuned together, which Step 8 demonstrates.

</details>

---

## Step 6 — Understand Feature Importance

Extract the feature importances from the gradient boosting model and print them ranked from most to least important.

<details>
<summary>Solution</summary>

```python
importances = gbr.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("=== Feature Importances ===")
for rank, idx in enumerate(sorted_idx, 1):
    print(f"  {rank:2}. {diabetes.feature_names[idx]:<10} importance={importances[idx]:.4f}")
print()
```

Output:

```
=== Feature Importances ===
   1. bmi        importance=0.3892
   2. s5         importance=0.2530
   3. bp         importance=0.0883
   4. s2         importance=0.0602
   5. s6         importance=0.0468
   6. age        importance=0.0444
   7. s1         importance=0.0376
   8. s3         importance=0.0366
   9. s4         importance=0.0318
  10. sex        importance=0.0122
```

Like random forest, gradient boosting's `feature_importances_` measures the mean decrease in impurity attributable to each feature, averaged across every tree and every split. `bmi` and `s5` (a blood serum measurement) dominate — consistent with BMI being a well-known clinical predictor of diabetes progression.

</details>

---

## Step 7 — Visualise

Display the following three plots:

- **7a** — A bar chart of RSS after each round of the from-scratch boosting loop (Step 3), showing the residual shrinking.
- **7b** — Train vs. test MSE at every boosting round of the full model (`gbr.staged_predict`), with a marker at the round with the lowest test MSE.
- **7c** — A horizontal bar chart of feature importances.

Do not save figures to disk — display them inline.

<details>
<summary>Solution</summary>

### 7a: Residual Shrinking (From-Scratch Loop)

```python
rss_by_round = [2_144_968, 1_425_438, 1_166_531, 1_030_110]  # from Step 3's printed output

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(len(rss_by_round)), rss_by_round, tick_label=[f"Round {i}" for i in range(len(rss_by_round))])
ax.set_ylabel("Residual sum of squares")
ax.set_title("RSS Shrinking Across 3 Hand-Built Boosting Rounds")
fig.tight_layout()
plt.show()
```

Each bar is shorter than the last — direct visual evidence that fitting a tree to the residuals and adding it back (scaled by `learning_rate`) removes real signal each round.

### 7b: Staged Train/Test Error

```python
train_mse_stages = []
test_mse_stages = []
for train_pred_stage, test_pred_stage in zip(gbr.staged_predict(X_train), gbr.staged_predict(X_test)):
    train_mse_stages.append(mean_squared_error(y_train, train_pred_stage))
    test_mse_stages.append(mean_squared_error(y_test, test_pred_stage))

best_round = int(np.argmin(test_mse_stages)) + 1

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
```

Train MSE keeps falling for all 200 rounds — the model can always fit the training residuals a little better by adding one more tree. Test MSE bottoms out much earlier (round 81 in this run) and then creeps back up: past that point, additional trees are fitting noise specific to the training set rather than generalizable signal. This is exactly the failure mode `n_iter_no_change` (early stopping) is designed to catch — a random forest has no equivalent curve, since averaging independent trees can't overfit this way.

### 7c: Feature Importance Bar Chart

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(range(len(importances)), importances[sorted_idx][::-1])
ax.set_yticks(range(len(importances)))
ax.set_yticklabels([diabetes.feature_names[i] for i in sorted_idx][::-1])
ax.set_xlabel("Importance (mean decrease in impurity)")
ax.set_title("Feature Importances — Gradient Boosting")
fig.tight_layout()
plt.show()
```

</details>

---

## Step 8 — Learning Rate vs. n_estimators Tradeoff

Train models across a grid of `learning_rate` ∈ {0.01, 0.05, 0.1, 0.3} × `n_estimators` ∈ {10, 25, 50, 100, 200, 400}. Record test R² for each combination and plot one curve per learning rate. In 2–3 sentences, describe the tradeoff between `learning_rate` and `n_estimators`.

<details>
<summary>Solution</summary>

```python
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
```

Output:

```
=== Learning Rate vs n_estimators ===
  lr=0.01  ['0.062', '0.153', '0.267', '0.389', '0.458', '0.481']
  lr=0.05  ['0.271', '0.418', '0.466', '0.475', '0.465', '0.440']
  lr=0.1   ['0.392', '0.466', '0.466', '0.455', '0.439', '0.397']
  lr=0.3   ['0.442', '0.417', '0.386', '0.312', '0.272', '0.256']
```

A low learning rate (`0.01`) needs hundreds of rounds just to catch up — at `n_estimators=10` it barely beats a constant prediction (R²=0.062) — but keeps improving all the way to 400 rounds, ending as the best score in the whole grid (R²=0.481, edging out random forest's 0.476 from Step 5). A high learning rate (`0.3`) makes fast progress early but overfits and *degrades* with more rounds, since each oversized correction increasingly chases training noise. The practical rule: pick the smallest `learning_rate` you can afford, and pair it with enough `n_estimators` (or early stopping) to let it converge — trading training time for a better-generalizing model.

</details>

---

## Step 9 — Make a Single Prediction (Inference)

Using the trained gradient boosting model, predict the disease progression score for the first sample in the test set. Print the predicted value alongside the true label.

<details>
<summary>Solution</summary>

```python
sample = X_test[[0]]
prediction = gbr.predict(sample)[0]
true_value = y_test[0]

print("=== Single Prediction ===")
print("Input features (first test sample):")
for name, val in zip(diabetes.feature_names, sample[0]):
    print(f"  {name:<10} {val:.4f}")
print(f"\nPredicted disease progression score: {prediction:.1f}")
print(f"True disease progression score:      {true_value:.1f}")
```

Output:

```
=== Single Prediction ===
Predicted disease progression score: 156.9
True disease progression score:      219.0
```

Unlike the classification tutorials, `predict` here returns a continuous value directly — there's no `predict_proba` step, since gradient boosting regression has no notion of class probability. The gap between predicted (156.9) and true (219.0) is a reminder that R²≈0.47–0.48 leaves real per-sample error: the diabetes dataset's 10 baseline features only partially explain disease progression a year later, and no amount of ensemble tuning fully closes that gap.

</details>
