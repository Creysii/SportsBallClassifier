"""
data_loader.py
--------------
Responsable único: recorrer la estructura de carpetas del dataset
(train/<clase>/*.jpg, test/<clase>/*.jpg) y devolver una lista de
tuplas (ruta_absoluta, etiqueta) lista para ser consumida por el
pipeline de extracción de características.

Se mantiene deliberadamente sin dependencias de OpenCV/sklearn: su
única responsabilidad es E/S sobre el sistema de archivos (principio
de responsabilidad única).
"""

from pathlib import Path
from typing import List, Tuple

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(split_dir: Path) -> List[Tuple[str, str]]:
    """
    Recorre split_dir (train/ o test/) asumiendo que cada subcarpeta
    directa representa una clase.

    Parameters
    ----------
    split_dir : Path
        Carpeta raíz del split (por ejemplo data/train).

    Returns
    -------
    List[Tuple[str, str]]
        Lista de (ruta_imagen_str, nombre_clase).
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta '{split_dir}'. "
            f"Verifica config.DATA_ROOT."
        )

    samples: List[Tuple[str, str]] = []
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"'{split_dir}' no contiene subcarpetas de clase.")

    for class_dir in class_dirs:
        label = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in VALID_EXTENSIONS:
                samples.append((str(img_path), label))

    return samples


def dataset_summary(samples: List[Tuple[str, str]]) -> dict:
    """Devuelve un conteo de imágenes por clase (útil para el reporte)."""
    summary = {}
    for _, label in samples:
        summary[label] = summary.get(label, 0) + 1
    return dict(sorted(summary.items()))


if __name__ == "__main__":
    # Prueba rápida manual: python -m src.data_loader
    import config
    train_samples = list_images(config.TRAIN_DIR)
    test_samples = list_images(config.TEST_DIR)
    print(f"Train: {len(train_samples)} imágenes")
    print(f"Test:  {len(test_samples)} imágenes")
    print("Distribución train:", dataset_summary(train_samples))
