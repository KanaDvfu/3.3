import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_PATH = "data/turkiye-student-evaluation_generic.csv"
OUT_TABLES = "outputs/tables"
OUT_FIGS = "outputs/figures"

os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_FIGS, exist_ok=True)

def run_pca_and_save(
    df: pd.DataFrame,
    features: list[str],
    tag: str,
    standardize: bool = True,
    title_suffix: str = ""
):
    # Выполняет PCA на признаках features, опционально стандартизует данные,
    # сохраняет таблицы explained variance, loadings и график cumulative variance.

    X = df[features].copy()

    # 1) стандартизация
    if standardize:
        scaler = StandardScaler()
        X_used = scaler.fit_transform(X)
        std_tag = "std"
    else:
        X_used = X.values
        std_tag = "raw"

    # 2) PCA
    pca = PCA()
    _ = pca.fit_transform(X_used)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # 3) Таблица explained variance
    variance_df = pd.DataFrame({
        "Component": np.arange(1, len(explained) + 1),
        "Explained Variance Ratio": explained,
        "Cumulative Explained Variance": cumulative
    })

    # 4) k для 80%
    k_80 = int(np.argmax(cumulative >= 0.8) + 1)

    # 5) коэффициенты компонент
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(features))]
    )

    # 6) Топ-10 по модулю весов PC1
    top_pc1_df = (
        loadings_df[["PC1"]]
        .assign(abs_PC1=lambda d: d["PC1"].abs())
        .sort_values("abs_PC1", ascending=False)
        .head(10)
    )

    variance_path = f"{OUT_TABLES}/pca_{tag}_{std_tag}_variance.csv"
    loadings_pc12_path = f"{OUT_TABLES}/pca_{tag}_{std_tag}_loadings_PC1_PC2.csv"
    top_pc1_path = f"{OUT_TABLES}/pca_{tag}_{std_tag}_top10_PC1.csv"

    variance_df.to_csv(variance_path, index=False)
    loadings_df[["PC1", "PC2"]].to_csv(loadings_pc12_path)
    top_pc1_df.to_csv(top_pc1_path)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
    plt.axhline(0.8, linestyle="--")  # линия 80%
    plt.axvline(k_80, linestyle="--")  # вертикаль на k_80
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA cumulative variance ({tag}, {std_tag}){title_suffix}")
    plt.grid()

    fig_path = f"{OUT_FIGS}/pca_{tag}_{std_tag}_cumulative.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Возвращаем полезные числа для текста отчёта
    return {
        "n_samples": int(df.shape[0]),
        "n_features": int(len(features)),
        "pc1": float(explained[0]),
        "k_80": k_80,
        "variance_path": variance_path,
        "loadings_pc12_path": loadings_pc12_path,
        "top_pc1_path": top_pc1_path,
        "fig_path": fig_path
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # признаки анкеты
    features = [c for c in df.columns if c.startswith("Q")]

    # PCA на всём наборе, стандартизовано
    res = run_pca_and_save(
        df=df,
        features=features,
        tag="all",
        standardize=True,
        title_suffix=" - All dataset"
    )

    print("=== PCA (2d) ALL DATASET, STANDARDIZED ===")
    print(f"Samples: {res['n_samples']}, Features: {res['n_features']}")
    print(f"PC1 explained variance ratio: {res['pc1']:.6f} ({res['pc1']*100:.2f}%)")
    print(f"Components needed for >=80% variance: {res['k_80']}")
    print("\nSaved files:")
    print(" - variance table:", res["variance_path"])
    print(" - loadings PC1-PC2:", res["loadings_pc12_path"])
    print(" - top10 PC1:", res["top_pc1_path"])
    print(" - cumulative plot:", res["fig_path"])


if __name__ == "__main__":
    main()