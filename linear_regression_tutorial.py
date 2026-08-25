"""
Workshop Tutorial: Linear Regression with scikit-learn + SHAP
==============================================================

Linear regression is a supervised learning algorithm that predicts a
continuous number (not a class label) as a weighted sum of the input
features:

    y_hat = w0 + w1*x1 + w2*x2 + ... + wn*xn

Training finds the weights (w0..wn) that minimise the sum of squared
residuals between predicted and actual values (Ordinary Least Squares).

This tutorial uses a tiny, hand-built house price dataset to keep every
number traceable, then covers four ways to look "inside" the model:
  1. SHAP (Shapley values)  — explain individual predictions
  2. R^2                    — how much variance the model explains overall
  3. VIF                    — whether the coefficients can be trusted
  4. OLS summary table      — coefficients, p-values and diagnostics at a glance
"""

# ------------------------------------------------------------
# Step 0 — Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------------------------------
# Step 1 — Build the dataset & fit the model
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Step 2 — Explain individual predictions with SHAP
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Step 3 — Visualise a prediction with a waterfall plot
# ------------------------------------------------------------
print("=== PART 3: Visualising a Prediction with a Waterfall Plot ===\n")

shap.plots.waterfall(sv1[0], max_display=5, show=False)
plt.title("SHAP Explanation - Typical House (125 m², 4 bedrooms)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Step 4 — Compute R^2 from scratch
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Step 5 — Check for multicollinearity with VIF
# ------------------------------------------------------------
print("=== PART 5: VIF - Multicollinearity Check ===\n")

X_const = sm.add_constant(X)  # Important: add intercept

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_const.values, i + 1)
    for i in range(X.shape[1])
]

print(vif_data.round(3))

# ------------------------------------------------------------
# Step 6 — The whole story in one table
# ------------------------------------------------------------
print("\n=== PART 6: Full Regression Summary ===\n")

# Same OLS fit as sklearn's LinearRegression, but statsmodels keeps the
# statistics around: standard errors, t-values, p-values, confidence intervals.
ols_model = sm.OLS(y, X_const).fit()  # X_const already has the intercept (Step 5)
print(ols_model.summary())
