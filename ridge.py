import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Fixed: changed from 'import matplotlib as plt'
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler  


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_prediction = linear_model.predict(X_test_scaled)


parameters = {'alpha': [0.01, 0.1, 1, 10, 100]}

ridge = Ridge()
ridge_cv = GridSearchCV(estimator=ridge, param_grid=parameters, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
ridge_prediction = ridge_cv.predict(X_test_scaled)


lasso = Lasso(max_iter=10000)
lasso_cv = GridSearchCV(estimator=lasso, param_grid=parameters, cv=5)
lasso_cv.fit(X_train_scaled, y_train)
lasso_prediction = lasso_cv.predict(X_test_scaled)

mse_linear = mean_squared_error(y_test, linear_prediction)
r2_linear = r2_score(y_test, linear_prediction)

mse_ridge = mean_squared_error(y_test, ridge_prediction)
r2_ridge = r2_score(y_test, ridge_prediction)

mse_lasso = mean_squared_error(y_test, lasso_prediction)
r2_lasso = r2_score(y_test, lasso_prediction)

best_ridge_alpha = ridge_cv.best_params_['alpha']
best_lasso_alpha = lasso_cv.best_params_['alpha']

print(f"Best Alpha for Ridge Regression: {best_ridge_alpha}")
print(f"Best Alpha for Lasso Regression: {best_lasso_alpha}")
print("\n" + "="*50 + "\n")

performance_data = {
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "Best Alpha": ["N/A", best_ridge_alpha, best_lasso_alpha],
    "MSE": [mse_linear, mse_ridge, mse_lasso],
    "R2 Score": [r2_linear, r2_ridge, r2_lasso]
}

comparison_df = pd.DataFrame(performance_data)

print("Model Performance Comparison Table:")
print(comparison_df.to_string(index=False))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
models = comparison_df["Model"]
mses = comparison_df["MSE"]
r2_scores = comparison_df["R2 Score"]


colors = ['#e74c3c', '#3498db', '#2ecc71']


bars1 = ax1.bar(models, mses, color=colors, edgecolor='black', width=0.5)
ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
ax1.set_title("Model Comparison: MSE ", fontsize=12, fontweight='bold', pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(0, max(mses) * 1.15) 


for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (max(mses) * 0.02), 
             f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')


bars2 = ax2.bar(models, r2_scores, color=colors, edgecolor='black', width=0.5)
ax2.set_ylabel("R² Score", fontsize=11)
ax2.set_title("Model Comparison: R² Score", fontsize=12, fontweight='bold', pad=15)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(0, max(r2_scores) * 1.15)  


for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max(r2_scores) * 0.02), 
             f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')


plt.tight_layout()
plt.show()