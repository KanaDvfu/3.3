import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- Пути ----------
DATA_PATH = "data/GiveMeSomeCredit-training.csv"
FIG_PATH = "outputs/figures"
TABLE_PATH = "outputs/tables"

os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)

# ---------- Загрузка данных ----------
df = pd.read_csv(DATA_PATH)

print("Размерность исходных данных:", df.shape)

# Удаляем пропуски
df = df.dropna()
print("Размерность после удаления NaN:", df.shape)

# Удаляем ID (если есть)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Отделяем целевую переменную
if "SeriousDlqin2yrs" in df.columns:
    y = df["SeriousDlqin2yrs"]
    X = df.drop(columns=["SeriousDlqin2yrs"])
else:
    X = df.copy()

print("Размерность матрицы признаков:", X.shape)

# ---------- Стандартизация ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- PCA ----------
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

# ---------- Сохранение таблицы ----------
results_df = pd.DataFrame({
    "Component": np.arange(1, len(explained_var)+1),
    "Explained Variance Ratio": explained_var,
    "Cumulative Variance": cumulative_var
})

results_df.to_csv(f"{TABLE_PATH}/pca_variance_results.csv", index=False)

print(results_df)

# ---------- График накопленной дисперсии ----------
plt.figure(figsize=(8,5))
plt.plot(np.arange(1, len(cumulative_var)+1), cumulative_var, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance (GiveMeSomeCredit)")
plt.grid()
plt.savefig(f"{FIG_PATH}/cumulative_variance.png", dpi=300)
plt.show()

# ---------- Сколько компонент объясняют 80% ----------
k_80 = np.argmax(cumulative_var >= 0.8) + 1
print(f"\nКоличество компонент для объяснения ≥80% дисперсии: {k_80}")