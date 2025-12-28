import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from Linear_Regression import LinearRegression as LR

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(
    "StudentPerformance.csv",
    usecols=["Previous Scores","Performance Index"]
).dropna()

X = df[["Previous Scores"]].values   # independent variable
Y = df["Performance Index"].values                  # dependent variable

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)

# -----------------------------
# Scale ONLY X
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = LR(learning_rate=0.05, num_of_iterations=50000)
model.fit(X_train_scaled, Y_train)

print("Weight:", model.w)
print("Bias:", model.b)

# -----------------------------
# Predict
# -----------------------------
Y_pred = model.predict(X_test_scaled)

# -----------------------------
# Evaluate
# -----------------------------
print("R2 Score:", r2_score(Y_test, Y_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_test, Y_pred)))

# -----------------------------
# Correct visualization
# -----------------------------
sorted_idx = np.argsort(X_test[:, 0])

plt.figure(figsize=(8, 5))
plt.scatter(X_test[:, 0], Y_test, color="red", label="Actual")
plt.plot(
    X_test[sorted_idx, 0],
    Y_pred[sorted_idx],
    color="blue",
    linewidth=3,
    label="Predicted"
)
plt.xlabel("Hours Studied ")
plt.ylabel("Performance Index")
plt.title("Performance Index vs Hours Studied (Linear Regression)")
plt.legend()
plt.show()
