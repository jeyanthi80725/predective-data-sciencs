import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Hours studied -> Pass (1) / Fail (0)
hours = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
                  5.5, 6, 6.5, 7, 7.5, 8]).reshape(-1, 1)
passed = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])

X_train, X_test, y_train, y_test = train_test_split(
    hours, passed, test_size=0.25, random_state=1, stratify=passed)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Probability of passing with 4 hours:",
      model.predict_proba([[4]])[0][1])
print("Prediction for 6 hours:", model.predict([[6]])[0])
