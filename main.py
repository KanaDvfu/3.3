import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# ----------------------------
# Пути
# ----------------------------
DATA_PATH = "data/turkiye-student-evaluation_generic.csv"
OUT_TABLES = "outputs/tables"
OUT_FIGS = "outputs/figures"

os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_FIGS, exist_ok=True)


# ----------------------------
# Утилита: PCA + сохранение
# ----------------------------
def run_pca_and_save_raw(df: pd.DataFrame, features: list[str], tag: str, title_suffix: str = ""):

    # PCA на НЕстандартизованных данных.
    # Сохраняет таблицы variance, компоненты (PC1-PC2), топ 10 PC1 и график cumulative.

    X = df[features].copy().values  # raw

    pca = PCA()
    _ = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    k_80 = int(np.argmax(cumulative >= 0.8) + 1)

    # таблицы variance
    variance_df = pd.DataFrame({
        "Component": np.arange(1, len(explained) + 1),
        "Explained Variance Ratio": explained,
        "Cumulative Explained Variance": cumulative
    })

    # компоненты
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(features))]
    )

    # топ 10 PC1
    top_pc1_df = (
        loadings_df[["PC1"]]
        .assign(abs_PC1=lambda d: d["PC1"].abs())
        .sort_values("abs_PC1", ascending=False)
        .head(10)
    )

    # сохраняем таблицы
    variance_path = f"{OUT_TABLES}/pca_{tag}_raw_variance.csv"
    loadings_pc12_path = f"{OUT_TABLES}/pca_{tag}_raw_loadings_PC1_PC2.csv"
    top_pc1_path = f"{OUT_TABLES}/pca_{tag}_raw_top10_PC1.csv"

    variance_df.to_csv(variance_path, index=False)
    loadings_df[["PC1", "PC2"]].to_csv(loadings_pc12_path)
    top_pc1_df.to_csv(top_pc1_path)

    # график cumulative
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
    plt.axhline(0.8, linestyle="--")
    plt.axvline(k_80, linestyle="--")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"PCA cumulative variance ({tag}, raw){title_suffix}")
    plt.grid()

    fig_path = f"{OUT_FIGS}/pca_{tag}_raw_cumulative.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return {
        "n_samples": int(df.shape[0]),
        "n_features": int(len(features)),
        "pc1": float(explained[0]),
        "k_80": k_80,
        "variance_path": variance_path,
        "loadings_pc12_path": loadings_pc12_path,
        "top_pc1_path": top_pc1_path,
        "fig_path": fig_path,
    }


# ----------------------------
# Выбор двух классов у instr=3
# ----------------------------
def pick_two_classes_for_instructor(df: pd.DataFrame, instr_id: int) -> list[int]:
    return [5, 8]


def main():
    df = pd.read_csv(DATA_PATH)
    features = [c for c in df.columns if c.startswith("Q")]

    print("=== 2e (RAW, no standardization) ===")

    # (b) one class = 5
    df_b = df[df["class"] == 5]
    res_b = run_pca_and_save_raw(df_b, features, tag="class5", title_suffix=" - class=5")
    print("\n(b) class=5")
    print(f"Samples: {res_b['n_samples']}")
    print(f"PC1: {res_b['pc1']:.6f} ({res_b['pc1']*100:.2f}%)")
    print(f"k80: {res_b['k_80']}")
    print("Saved:", res_b["fig_path"])

    # (c) instr = 3, two classes auto-picked
    instr_id = 3
    c1, c2 = pick_two_classes_for_instructor(df, instr_id)
    df_c = df[(df["instr"] == instr_id) & (df["class"].isin([c1, c2]))]
    res_c = run_pca_and_save_raw(df_c, features, tag=f"instr{instr_id}_classes{c1}_{c2}",
                                 title_suffix=f" - instr={instr_id}, classes={c1},{c2}")
    print(f"\n(c) instr={instr_id}, classes={c1},{c2}")
    print(f"Samples: {res_c['n_samples']}")
    print(f"PC1: {res_c['pc1']:.6f} ({res_c['pc1']*100:.2f}%)")
    print(f"k80: {res_c['k_80']}")
    print("Saved:", res_c["fig_path"])

    # (d) all dataset
    res_d = run_pca_and_save_raw(df, features, tag="all", title_suffix=" - All dataset")
    print("\n(d) all dataset")
    print(f"Samples: {res_d['n_samples']}")
    print(f"PC1: {res_d['pc1']:.6f} ({res_d['pc1']*100:.2f}%)")
    print(f"k80: {res_d['k_80']}")
    print("Saved:", res_d["fig_path"])

    print("\nSaved tables are in outputs/tables/, figures in outputs/figures/.")


if __name__ == "__main__":
    main()