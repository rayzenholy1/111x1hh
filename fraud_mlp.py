import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(7)

n_legit = 400
n_fraud = 100

legit = pd.DataFrame({
    "amount": np.random.exponential(2.0, n_legit),
    "failed_logins": np.random.poisson(0.3, n_legit),
    "distance": np.random.exponential(0.5, n_legit),
    "target": 0
})

fraud = pd.DataFrame({
    "amount": np.random.uniform(5, 40, n_fraud),
    "failed_logins": np.random.randint(2, 8, n_fraud),
    "distance": np.random.uniform(3, 30, n_fraud),
    "target": 1
})

data = pd.concat([legit, fraud], ignore_index=True)
data = data.sample(frac=1, random_state=1).reset_index(drop=True)

print("Перші рядки датасету:")
print(data.head())
print("\nРозподіл класів:")
print(data["target"].value_counts())

X = data[["amount", "failed_logins", "distance"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

pred = model.predict(X_test_s)

print("\nТочність:", round(accuracy_score(y_test, pred), 4))
print("\nМатриця помилок:")
print(confusion_matrix(y_test, pred))
print("\nЗвіт класифікації:")
print(classification_report(y_test, pred, target_names=["Легітимна", "Шахрайська"]))

new_tx = pd.DataFrame({
    "amount": [1.5, 25.0, 3.0],
    "failed_logins": [0, 5, 1],
    "distance": [0.2, 15.0, 1.0]
})
new_pred = model.predict(scaler.transform(new_tx))
new_proba = model.predict_proba(scaler.transform(new_tx))[:, 1]

print("Перевірка нових транзакцій:")
for i in range(len(new_tx)):
    row = new_tx.iloc[i]
    status = "ШАХРАЙСЬКА" if new_pred[i] == 1 else "легітимна"
    print(f"Сума: {row['amount']} тис. грн, невдалих входів: {int(row['failed_logins'])}, відстань: {row['distance']} сотень км -> {status} (ймовірність шахрайства: {new_proba[i]:.2f})")

fig, axs = plt.subplots(1, 2, figsize=(13, 5))

axs[0].plot(model.loss_curve_, color="purple")
axs[0].set_xlabel("Ітерація")
axs[0].set_ylabel("Втрати")
axs[0].set_title("Крива навчання")
axs[0].grid(True)

ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Легітимна", "Шахрайська"]).plot(ax=axs[1], cmap="Blues", colorbar=False)
axs[1].set_title("Матриця помилок")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data[data["target"] == 0]["amount"], data[data["target"] == 0]["distance"], color="blue", alpha=0.5, label="Легітимна")
plt.scatter(data[data["target"] == 1]["amount"], data[data["target"] == 1]["distance"], color="red", alpha=0.7, label="Шахрайська")
plt.xlabel("Сума транзакції, тис. грн")
plt.ylabel("Відстань від звичної геолокації, сотні км")
plt.title("Розподіл транзакцій за класами")
plt.legend()
plt.grid(True)
plt.show()
