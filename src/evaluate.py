"""
evaluate.py
-----------
Genera las métricas obligatorias de la sección de Resultados del
reporte: accuracy por modelo, matriz de confusión por modelo (guardada
como figura), y una tabla comparativa final ordenada por desempeño.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin display, necesario en entornos headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import config


def evaluate_predictions(model_name: str, y_true, y_pred, class_names,
                          save_dir: Path = config.FIGURES_DIR) -> dict:
    """
    Calcula accuracy y matriz de confusión para un modelo, guarda la
    matriz como imagen PNG y devuelve un resumen en dict.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    report = classification_report(y_true, y_pred, labels=class_names,
                                    output_dict=True, zero_division=0)

    _plot_confusion_matrix(cm, class_names, model_name, acc,
                            save_dir / f"confusion_matrix_{model_name}.png")

    # Guarda también la matriz cruda en CSV, útil para incluirla como
    # tabla en el documento Word/LaTeX del reporte.
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(save_dir / f"confusion_matrix_{model_name}.csv")

    print(f"[evaluate] {model_name}: accuracy = {acc:.4f}")

    return {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def _plot_confusion_matrix(cm, class_names, model_name, acc, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Matriz de Confusión - {model_name} (Accuracy={acc:.3f})")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6, color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_comparison_table(results: list, save_dir: Path = config.FIGURES_DIR) -> pd.DataFrame:
    """
    Construye y guarda la tabla comparativa de Accuracy/F1 de todos los
    modelos, ordenada de mejor a peor. Esta es la tabla central exigida
    por la rúbrica en "Resultados: ... Accuracy (Precisión global)
    comparativo de todos los modelos implementados".
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(save_dir / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["model"], df["accuracy"], color="#4C72B0")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Comparación de Accuracy por Modelo")
    for i, v in enumerate(df["accuracy"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    print("\n=== Tabla comparativa final ===")
    print(df.to_string(index=False))

    return df
