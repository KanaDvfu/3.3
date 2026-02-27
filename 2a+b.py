import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
df = pd.read_csv("data/turkiye-student-evaluation_generic.csv")

# Выбран класс 5
_class = 5
df_class = df[df["class"] == _class]
print(df_class.shape[0])

# Только вопросы
features = [col for col in df.columns if col.startswith("Q")]
X = df_class[features]

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

# Таблица
results = pd.DataFrame({
    "Component": np.arange(1, len(explained)+1),
    "Explained Variance": explained,
    "Cumulative Variance": cumulative
})

results.to_csv("outputs/tables/pca_class1_variance.csv", index=False)
print(results)

# График
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative)+1), cumulative, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - One Class (Standardized)")
plt.grid()

plt.savefig("outputs/figures/pca_class1_cumulative.png", dpi=300)
plt.close()

# Сколько компонент для 80%
k_80 = np.argmax(cumulative >= 0.8) + 1
print("Компонент для 80%:", k_80)

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(features))],
    index=features
)

loadings.to_csv("outputs/tables/pca_class1_loadings.csv")
print(loadings[["PC1", "PC2"]].sort_values(by="PC1", key=abs, ascending=False).head(10))