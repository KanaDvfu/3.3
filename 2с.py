import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
df = pd.read_csv("data/turkiye-student-evaluation_generic.csv")

# Выбор преподавателя
df_instr = df[df["instr"] == 3]

# Узнаем его предметы
print(df_instr["class"].unique())

# Выбираем два предмета (например 5 и 8)
selected_classes = [5, 8]

df_two = df_instr[df_instr["class"].isin(selected_classes)]

print("Размер выборки:", df_two.shape)

# Только вопросы
features = [col for col in df.columns if col.startswith("Q")]
X = df_two[features]

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

k_80 = np.argmax(cumulative >= 0.8) + 1

print("PC1:", explained[0])
print("Компонент для 80%:", k_80)

# ----- Анализ весов первой главной компоненты (PC1) -----
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    index=features
)

pc1_weights = loadings["PC1"].sort_values(key=abs, ascending=False)

print("\n--- Веса первой главной компоненты (PC1) ---")
print("Наибольший вклад признаков в PC1:")
print(pc1_weights.head(10))

# Сохраняем веса PC1
pc1_weights_df = pc1_weights.reset_index()
pc1_weights_df.columns = ["Feature", "PC1_weight"]
pc1_weights_df.to_csv("outputs/tables/pca_instr3_two_classes_pc1_weights_sorted.csv", index=False)

print("\nСумма модулей весов PC1:", np.sum(np.abs(loadings["PC1"])))
print("Если большинство весов одного знака, PC1 отражает общий фактор (например, общую оценку курса).")

pc2_weights = loadings["PC2"].sort_values(key=abs, ascending=False)

print("\n--- Веса первой главной компоненты (PC2) ---")
print("Наибольший вклад признаков в PC2:")
print(pc2_weights.head(10))

print("\nСумма модулей весов PC2:", np.sum(np.abs(loadings["PC2"])))
print("Если большинство весов одного знака, PC1 отражает общий фактор (например, общую оценку курса).")

# Таблица
results_two = pd.DataFrame({
    "Component": np.arange(1, len(explained)+1),
    "Explained Variance": explained,
    "Cumulative Variance": cumulative
})
results_two.to_csv("outputs/tables/pca_instr3_two_classes_variance.csv", index=False)

# График
plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative)+1), cumulative, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Two Classes (Instructor 3)")
plt.grid()
plt.savefig("outputs/figures/pca_instr3_two_classes_cumulative.png", dpi=300)
plt.close()