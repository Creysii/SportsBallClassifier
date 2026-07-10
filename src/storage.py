"""
storage.py
----------
Estructura de datos para el almacenamiento de características
(requisito de la rúbrica: "Estructura de datos utilizada para el
almacenamiento (JSON o SQL)").

Se eligió JSON en lugar de SQL por tres razones justificadas en el
reporte técnico:
  1. Cada registro es un documento auto-contenido de longitud variable
     (ruta, clase, y varios vectores de distinta dimensionalidad por
     descriptor) — un esquema documental encaja mejor que una tabla
     relacional rígida.
  2. El dataset se procesa como un batch offline (no hay escrituras
     concurrentes ni consultas transaccionales), por lo que no se
     necesitan las garantías ACID de una base de datos SQL.
  3. Facilita la portabilidad: el archivo de features puede excluir
     directamente el pipeline pesado de extracción reutilizándose con
     pandas.read_json en cualquier máquina, sin necesidad de un motor
     de base de datos instalado.

Cada entrada almacenada tiene el esquema:
{
    "image_path": str,
    "label": str,
    "features": {
        "lbp": [float, ...],
        "hog": [float, ...],
        "hu": [float, ...],
        "freeman": [float, ...],
        "combined": [float, ...]
    }
}
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def save_features_json(records: List[dict], out_path: Path) -> None:
    """Guarda una lista de registros de características como JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for r in records:
        serializable.append({
            "image_path": r["image_path"],
            "label": r["label"],
            "features": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in r["features"].items()
            },
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)

    print(f"[storage] Guardadas {len(serializable)} entradas en {out_path}")


def load_features_json(in_path: Path) -> List[dict]:
    """Carga y reconstruye los registros de características desde JSON."""
    in_path = Path(in_path)
    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for r in raw:
        records.append({
            "image_path": r["image_path"],
            "label": r["label"],
            "features": {
                k: np.array(v, dtype="float32") for k, v in r["features"].items()
            },
        })
    return records


def to_matrix(records: List[dict], feature_key: str = "combined"):
    """
    Convierte una lista de registros en (X, y) listos para scikit-learn.

    Parameters
    ----------
    feature_key : str
        Cuál descriptor usar como matriz de diseño: 'combined' (default),
        'lbp', 'hog', 'hu' o 'freeman'. Útil para el análisis comparativo
        de "qué descriptor aporta más valor" (sección de Conclusiones).
    """
    X = np.stack([r["features"][feature_key] for r in records])
    y = np.array([r["label"] for r in records])
    return X, y
