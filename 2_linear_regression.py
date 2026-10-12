import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: years of experience vs salary (in thousands)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([30, 35, 41, 45, 52, 58, 61, 68, 73, 80])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))
print("Predicted salary for 11 years:", model.predict([[11]])[0])

plt.scatter(X, y, color="blue", label="Actual")
plt.plot(X, model.predict(X), color="red", label="Regression line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (thousands)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
