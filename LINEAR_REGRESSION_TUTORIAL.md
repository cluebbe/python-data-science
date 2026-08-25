# Linear Regression + SHAP — Step-by-Step Tutorial

## Introduction to Linear Regression

**Linear regression** is a supervised learning algorithm — like the decision trees, logistic regression, and neural networks elsewhere in this series — but it predicts a **continuous number** instead of a class label. Given a house's size, bedroom count, and age, it predicts a price in euros, not a category.

### How it works

The model learns one weight per feature plus an intercept, and predicts a weighted sum:

```
y_hat = w0 + w1·x1 + w2·x2 + ... + wn·xn
```

Training finds the weights that minimise the **sum of squared residuals** — the total squared distance between each actual value and the model's prediction — a method called **Ordinary Least Squares (OLS)**. Unlike logistic regression, there's no sigmoid squashing the output into a probability: the raw weighted sum *is* the prediction.

### Reading the coefficients

Each learned coefficient `wj` is the model's estimate of "how much does the prediction change for a one-unit increase in feature `j`, holding every other feature constant?" This makes linear regression one of the most directly interpretable models available — assuming its assumptions hold (see the VIF section below for one that often doesn't).

### Regression vs. classification

| | Classification (Decision Tree, Logistic Regression) | Regression (Linear Regression) |
|---|---|---|
| Predicts | A class label (e.g. benign/malignant) | A continuous number (e.g. price) |
| Output range | Discrete set of classes | Any real number |
| Typical evaluation | Accuracy, ROC-AUC | R², mean squared error |

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Highly interpretable — coefficients have direct real-world units | Assumes a linear relationship between features and target |
| Fast to train, even on large datasets | Sensitive to outliers (squared residuals penalise them heavily) |
| No hyperparameters to tune for the basic model | Coefficients become unreliable when features are correlated with each other (multicollinearity) |
| Well-understood statistical theory (confidence intervals, p-values) | Can't capture non-linear or interaction effects on its own |

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
mkdir linear-regression && cd linear-regression

# 2. Create a virtual environment (run once)
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install numpy pandas scikit-learn shap matplotlib statsmodels

# 5. When you're done, deactivate
deactivate
```

> **A note on `shap` and very new Python versions:** `shap` depends on `numba` for some of its internals, and `numba`'s compiler backend (`llvmlite`) typically takes a few months to add support for a brand-new Python release. If `pip install shap` fails while building `llvmlite`, the fix isn't in your code — use a Python version one or two minor releases behind the latest (e.g. 3.12 or 3.13) for this tutorial's virtual environment.

---

## Preparation — Imports

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

- **numpy** — array math and manual metric calculations
- **pandas** — building the dataset as a labelled `DataFrame`
- **sklearn.linear_model** — the `LinearRegression` model
- **shap** — Shapley-value explanations for individual predictions
- **matplotlib** — rendering the SHAP waterfall plot
- **statsmodels** — the Variance Inflation Factor multicollinearity check (not available in scikit-learn)

---

## Step 1 — Build the Dataset & Fit the Model

Build a small `DataFrame` with 10 houses and four columns: `size_m2`, `bedrooms`, `age_years`, and the target `price_kEUR`. Split it into features `X` (the first three columns) and target `y`. Fit a `LinearRegression` model and print its three coefficients.

<details>
<summary>Solution</summary>

```python
print("=== PART 1: Simple Linear Regression ===\n")

data = {
    "size_m2":    [75, 82, 90, 98, 105, 112, 120, 128, 135, 142],
    "bedrooms":   [ 3,  2,  4,  4,   2,   2,   4,   4,   5,   5],
    "age_years":  [28,  5, 22, 12,  31,   8,  25,  15,   3,  19],
    "price_kEUR": [268, 299, 321, 387, 277, 355, 412, 421, 506, 507],
}

df = pd.DataFrame(data)
print("Our tiny dataset:")
print(df)
print("\n")

X = df[["size_m2", "bedrooms", "age_years"]]
y = df["price_kEUR"]

model = LinearRegression()
model.fit(X, y)

print("Model learned these coefficients:")
print(f"Size (per m²)              : {model.coef_[0]:.2f} kEUR")
print(f"Bedrooms (per extra room)  : {model.coef_[1]:.2f} kEUR")
print(f"Age (per year)             : {model.coef_[2]:.2f} kEUR")
print("\n")
```

**Why such a tiny, hand-built dataset?** With only 10 rows and 3 features, every intermediate number in the rest of this tutorial — a SHAP value, an R², a VIF — is small enough to sanity-check by hand or with a calculator. That traceability is worth more here than a "realistic" dataset size.

**Reading the coefficients:** `size_m2 ≈ 2.18` means each extra square meter adds about 2,180 EUR to the predicted price, holding bedrooms and age fixed. `age_years` is negative — older houses are predicted cheaper, all else equal. These numbers only mean "holding everything else constant" if the features aren't themselves correlated — which Step 5 (VIF) checks directly.

</details>

---

## Step 2 — Explain Individual Predictions with SHAP

Create a `shap.LinearExplainer` from the fitted model and the training features. Predict the price for two new houses — a "typical" large house (125 m², 4 bedrooms, 5 years old) and an "atypical" one (140 m², only 2 bedrooms, 20 years old) — and print each feature's SHAP contribution alongside the final prediction.

<details>
<summary>Solution</summary>

```python
print("=== PART 2: Explaining predictions with SHAP ===\n")

# LinearExplainer computes exact Shapley values for linear models —
# no sampling/approximation needed, unlike tree or neural network models.
explainer = shap.LinearExplainer(model, X)

# --- Example 1: typical large house ---
new_house1 = pd.DataFrame({"size_m2": [125], "bedrooms": [4], "age_years": [5]})
pred1 = model.predict(new_house1)[0]
sv1 = explainer(new_house1)

print("House 1 -> 125 m² with 4 bedrooms")
print(f"Predicted price: {pred1:.1f} kEUR")
print(f"SHAP size_m2   : {sv1.values[0][0]:+.2f} kEUR")
print(f"SHAP bedrooms  : {sv1.values[0][1]:+.2f} kEUR")
print(f"SHAP age       : {sv1.values[0][2]:+.2f} kEUR\n")

# --- Example 2: atypical house (large but few bedrooms) ---
new_house2 = pd.DataFrame({"size_m2": [140], "bedrooms": [2], "age_years": [20]})
pred2 = model.predict(new_house2)[0]
sv2 = explainer(new_house2)

print("House 2 -> 140 m² with only 2 bedrooms")
print(f"Predicted price: {pred2:.1f} kEUR")
print(f"SHAP size_m2   : {sv2.values[0][0]:+.2f} kEUR")
print(f"SHAP bedrooms  : {sv2.values[0][1]:+.2f} kEUR")
print(f"SHAP age       : {sv2.values[0][2]:+.2f} kEUR\n")
```

**What a Shapley value means:** `sv1.values[0][j]` is how much feature `j` pushed *this specific prediction* away from the model's average prediction (the "base value"). Positive means it pushed the price up; negative means down. Unlike the raw coefficients from Step 1 — which describe the model everywhere — Shapley values describe one prediction, in the units of the target (kEUR here), which is why they're the natural tool for explaining a single result to a non-technical stakeholder ("this house is priced 68 kEUR higher than average because of its size").

**Why `LinearExplainer` needs no approximation:** For a linear model, the Shapley value of feature `j` has a closed form: `φⱼ = wⱼ · (xⱼ − x̄ⱼ)`, where `x̄ⱼ` is that feature's average value across the training data (`X`, passed to the explainer as the "background" distribution). Summing all `φⱼ` plus the base value exactly reconstructs the prediction — nothing is estimated or sampled, which is why linear/logistic models get a dedicated, exact SHAP explainer while tree ensembles and neural networks need approximate ones.

**Comparing the two houses:** House 1 is "typical" for its size (4 bedrooms matches what a 125 m² house usually has), so `bedrooms` contributes a modest positive push. House 2 is large but under-bedroomed for its size, so `bedrooms` contributes a large *negative* push that partially cancels out the positive push from `size_m2` — the SHAP values expose that tension directly, whereas the raw coefficients alone would not tell you two features are pulling against each other for a specific house.

</details>

---

## Step 3 — Visualise a Prediction with a Waterfall Plot

Using the SHAP values for House 1 from Step 2, produce a waterfall plot. In 1–2 sentences, describe what the plot shows.

<details>
<summary>Solution</summary>

```python
print("=== PART 3: Visualising a Prediction with a Waterfall Plot ===\n")

shap.plots.waterfall(sv1[0], max_display=5, show=False)
plt.title("SHAP Explanation - Typical House (125 m², 4 bedrooms)")
plt.tight_layout()
plt.show()
```

**Reading the waterfall plot:** it starts at the base value (the model's average prediction across the training data, shown at the bottom) and stacks each feature's SHAP value as a bar — red bars push the prediction up, blue bars push it down — ending at the final predicted price at the top. It turns the printed numbers from Step 2 into a single narrative: "starting from the average house price, the size added X, bedrooms added Y, age added Z, landing on this house's specific prediction."

**`max_display=5`** caps how many feature bars are drawn individually; with only 3 features here it has no effect, but on a wider dataset it groups the smallest contributors into a single "other features" bar to keep the plot readable.

</details>

---

## Step 4 — Compute R² From Scratch

Compute the R² score (coefficient of determination) manually: the mean of `y`, the total sum of squares (`ss_total`), the explained sum of squares (`ss_explained`), and their ratio. Print all three values and a one-sentence interpretation of the result.

<details>
<summary>Solution</summary>

```python
print("=== PART 4: Calculating R² Score (Coefficient of Determination) ===\n")

y_mean = np.mean(y)
ss_total = np.sum((y - y_mean) ** 2)

y_pred = model.predict(X)
ss_explained = np.sum((y_pred - y_mean) ** 2)

r2 = ss_explained / ss_total

print(f"SS_total (Variance of actual data)    : {ss_total:.2f}")
print(f"SS_explained (Variance of predictions): {ss_explained:.2f}")
print(f"R² Score                               : {r2:.4f}")
print(f"-> The model explains {r2*100:.1f}% of the variation in house prices.\n")
```

**`ss_total`** measures how spread out the actual prices are around their mean — the variance the model is trying to explain. **`ss_explained`** measures how spread out the model's *predictions* are around that same mean — the variance the model actually captured.

**Why dividing `ss_explained` by `ss_total` works here:** the more familiar textbook formula is `R² = 1 − SS_residual/SS_total`, using the *error* the model didn't explain rather than the variance it did. The two formulas agree only when the residuals and predictions are uncorrelated — which is guaranteed for an OLS model fit with an intercept on its own training data (a Pythagorean-style decomposition: `SS_total = SS_explained + SS_residual` exactly). That's confirmed by comparing this result to scikit-learn's built-in score: `model.score(X, y)` returns the same value.

**Interpretation:** R² ranges from 1.0 (predictions perfectly match reality) down to 0.0 (the model does no better than always predicting the mean price), and can even go negative on new data if the model performs worse than that baseline. An R² of ~0.98 here means the three features together explain almost all of the price variation in this dataset — expected, since the data was constructed with a clear size/bedroom/age relationship rather than being noisy, real-world data.

</details>

---

## Step 5 — Check for Multicollinearity with VIF

Using `statsmodels`, add an intercept column to `X` with `sm.add_constant`, then compute the **Variance Inflation Factor (VIF)** for each of the three features. Print the results as a small table and interpret them.

<details>
<summary>Solution</summary>

```python
print("=== PART 5: VIF - Multicollinearity Check ===\n")

X_const = sm.add_constant(X)  # Important: add intercept

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_const.values, i + 1)
    for i in range(X.shape[1])
]

print(vif_data.round(3))
```

**What VIF measures:** for each feature, VIF fits a *separate* regression predicting that feature from all the *other* features, and reports `VIF = 1 / (1 − R²ⱼ)`. If a feature can be predicted almost perfectly from the others (e.g. `bedrooms` is almost a linear function of `size_m2`), its VIF is large — the model can no longer tell which of the correlated features actually deserves credit for the price, so both coefficients become unstable even though the model's overall predictions (and R²) can still look fine.

**Why `sm.add_constant` first:** VIF's internal regressions need an intercept to be meaningful — without one, the calculation implicitly forces the fitted line through the origin, which inflates every VIF and makes the values impossible to compare against the standard thresholds. `i + 1` in the loop skips column 0 (the constant itself), since VIF is only computed for the real features.

**Interpreting the numbers:**

| VIF | Interpretation |
|---|---|
| < 5 | Acceptable — little to no multicollinearity concern |
| 5 – 10 | Moderate — worth investigating, especially if coefficients look surprising |
| ≥ 10 | Severe — coefficients for these features are unreliable and shouldn't be interpreted individually |

For this dataset all three VIFs come out well under 5, so `size_m2`, `bedrooms`, and `age_years` are each contributing independent information — the coefficients from Step 1 and the SHAP values from Step 2 can be trusted. The broader lesson: **a high R² alone doesn't guarantee reliable feature interpretations** — always pair a performance metric with a diagnostic like VIF before trusting what the coefficients (or their SHAP values) say about *why* the model predicts what it does.

</details>

---

## Step 6 — Read the Full Regression Table

Everything so far was computed piece by piece: the coefficients in Step 1, R² in Step 4, a collinearity check in Step 5. `statsmodels` can produce all of it — plus the statistics scikit-learn never exposes — in a single table.

Fit an OLS model with `sm.OLS` on `y` and the `X_const` you built in Step 5, then print its `.summary()`. Compare the `coef` column against the coefficients you printed in Step 1 and the `R-squared` value against your Step 4 calculation.

<details>
<summary>Solution</summary>

```python
print("\n=== PART 6: Full Regression Summary ===\n")

# Same OLS fit as sklearn's LinearRegression, but statsmodels keeps the
# statistics around: standard errors, t-values, p-values, confidence intervals.
ols_model = sm.OLS(y, X_const).fit()  # X_const already has the intercept (Step 5)
print(ols_model.summary())
```

Expected output:

```
                            OLS Regression Results
==============================================================================
Dep. Variable:             price_kEUR   R-squared:                       0.981
Model:                            OLS   Adj. R-squared:                  0.971
Method:                 Least Squares   F-statistic:                     102.6
No. Observations:                  10   Prob (F-statistic):           1.52e-05
Df Residuals:                       6   Log-Likelihood:                -38.556
Df Model:                           3   AIC:                             85.11
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         58.7851     27.677      2.124      0.078      -8.939     126.509
size_m2        2.1835      0.280      7.812      0.000       1.500       2.867
bedrooms      33.2269      5.288      6.284      0.001      20.289      46.165
age_years     -2.2098      0.516     -4.286      0.005      -3.471      -0.948
==============================================================================
Omnibus:                        1.793   Durbin-Watson:                   2.533
Prob(Omnibus):                  0.408   Jarque-Bera (JB):                0.964
Skew:                          -0.406   Prob(JB):                        0.618
Kurtosis:                       1.714   Cond. No.                         664.
==============================================================================
```

**Why scikit-learn has no equivalent:** `LinearRegression` is built for *prediction* — it exposes `coef_`, `intercept_`, and `predict()`, and nothing else, because that's all a prediction pipeline needs. `statsmodels` comes from the statistics tradition instead, where the point of fitting a model is to make *inferences* about the coefficients. Both fit the identical OLS model to identical data, which is why `coef` here reproduces Step 1's numbers (2.18 / 33.23 / −2.21) exactly, and `R-squared` reproduces the 0.981 you derived by hand in Step 4.

**The columns that are genuinely new:**

| Column | What it tells you |
|---|---|
| `std err` | How much the coefficient estimate would wobble across different samples of data |
| `t` | The coefficient divided by its standard error — how many standard errors it sits from zero |
| `P>\|t\|` | The probability of seeing a coefficient this large if the feature's true effect were zero |
| `[0.025 0.975]` | The 95% confidence interval — the plausible range for the true coefficient |

**Reading the p-values:** `size_m2` (p ≈ 0.000), `bedrooms` (p = 0.001) and `age_years` (p = 0.005) are all comfortably below the conventional 0.05 threshold, so each is statistically distinguishable from "no effect". The intercept is not (p = 0.078) — unsurprising, since a 0 m², 0-bedroom house is far outside the data, and the intercept is the model extrapolating to a point it has never seen. **This is the one question SHAP and R² cannot answer.** SHAP tells you how much a feature moved a particular prediction; R² tells you how well the model fits overall; neither tells you whether a coefficient is distinguishable from noise. With only 10 rows, that distinction matters.

**Adjusted R² vs R²:** plain R² can only go up when you add a feature, even a useless random one, because extra columns always let OLS fit the training data a little better. Adjusted R² (0.971 here, against 0.981) penalises each added feature and can go *down*, which makes it the honest number when comparing models with different feature counts.

**F-statistic:** where each p-value tests one coefficient, the F-statistic tests the whole model at once — "do these three features *jointly* explain anything?" `Prob (F-statistic) = 1.52e-05` says yes, decisively.

**`Cond. No.` — a second opinion on Step 5:** the condition number is another multicollinearity signal, derived from the geometry of the feature matrix rather than from per-feature regressions like VIF. Values above ~1000 are the usual warning sign; 664 here is elevated mostly because the features live on wildly different scales (`size_m2` in the hundreds, `bedrooms` in single digits) rather than because they're redundant — which is exactly what the low VIFs in Step 5 already told you. Two diagnostics, same conclusion: these coefficients can be trusted.

**The residual diagnostics** (Omnibus, Jarque-Bera, Skew, Kurtosis) all test one assumption: that the residuals are normally distributed, which is what the p-values and confidence intervals above depend on. Both `Prob(Omnibus) = 0.408` and `Prob(JB) = 0.618` are well above 0.05, meaning there's no evidence against normality — though with 10 data points these tests have very little power to detect a problem even if one existed. **Durbin-Watson** (2.533) checks whether consecutive residuals are correlated; values near 2 indicate independence, which matters most for time-ordered data.

</details>
