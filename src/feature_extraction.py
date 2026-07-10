"""
feature_extraction.py
----------------------
Pipeline de extracción de características clásicas (Feature Extraction)
para el problema de clasificación de pelotas deportivas.

Se implementan cuatro descriptores, tal como exige la rúbrica:

1. LBP (Local Binary Pattern)   -> textura local
2. HOG (Histogram of Oriented Gradients) -> forma/gradientes globales
3. Momentos Invariantes de Hu   -> forma global invariante a
   traslación, escala y rotación
4. Código de Freeman (Freeman Chain Code) -> descriptor de contorno,
   se resume como un histograma normalizado de las 8 direcciones de
   8-conectividad (invariante a la posición del contorno)

Los cuatro vectores se concatenan en un único vector de características
por imagen (`extract_features`), y adicionalmente se exponen por
separado para poder analizarlos/graficarlos de forma independiente en
el reporte técnico.
"""

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

import config


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _load_and_resize(image_path: str) -> np.ndarray:
    """Carga una imagen a color y la redimensiona a IMG_SIZE."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    img = cv2.resize(img, config.IMG_SIZE, interpolation=cv2.INTER_AREA)
    return img


def _binary_mask(gray: np.ndarray) -> np.ndarray:
    """
    Segmenta el objeto principal (la pelota) usando umbralización de Otsu.
    Se usa tanto para Hu Moments como para el Código de Freeman, ya que
    ambos requieren un contorno/silueta bien definida del objeto.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Heurística: si el fondo quedó como "objeto" (más del 60% de píxeles
    # blancos), se invierte la máscara, asumiendo que la pelota ocupa
    # la minoría del encuadre.
    if np.mean(mask == 255) > 0.6:
        mask = cv2.bitwise_not(mask)
    return mask


# ---------------------------------------------------------------------------
# 1. Local Binary Pattern (LBP)
# ---------------------------------------------------------------------------
def extract_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Calcula el histograma normalizado de LBP uniforme.
    Longitud del vector: n_points + 2 (para method='uniform').
    """
    lbp = local_binary_pattern(
        gray, P=config.LBP_N_POINTS, R=config.LBP_RADIUS, method=config.LBP_METHOD
    )
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-7)  # normalización L1
    return hist


# ---------------------------------------------------------------------------
# 2. Histogram of Oriented Gradients (HOG)
# ---------------------------------------------------------------------------
def extract_hog(gray: np.ndarray) -> np.ndarray:
    """
    Calcula el descriptor HOG estándar sobre la imagen en escala de grises.
    """
    features = hog(
        gray,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype("float32")


# ---------------------------------------------------------------------------
# 3. Momentos Invariantes de Hu
# ---------------------------------------------------------------------------
def extract_hu_moments(mask: np.ndarray) -> np.ndarray:
    """
    Calcula los 7 Momentos Invariantes de Hu a partir de la máscara binaria.
    Se aplica una transformación log-signed para llevar los valores (que
    varían en varios órdenes de magnitud) a una escala comparable, tal
    como se recomienda en la literatura (Hu, 1962).
    """
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    # log-transform preservando el signo: -sign(h) * log10(|h|)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log.astype("float32")


# ---------------------------------------------------------------------------
# 4. Código de Freeman (Freeman Chain Code)
# ---------------------------------------------------------------------------
# Direcciones de 8-conectividad estándar (0 = Este, sentido antihorario)
_FREEMAN_DIRS = np.array([
    [1, 0], [1, -1], [0, -1], [-1, -1],
    [-1, 0], [-1, 1], [0, 1], [1, 1],
])


def extract_freeman(mask: np.ndarray) -> np.ndarray:
    """
    Extrae el contorno principal de la máscara y lo recorre como una
    cadena de Freeman de 8 direcciones. Como el código crudo tiene
    longitud variable (depende del perímetro del objeto), se resume
    como un histograma normalizado de frecuencia de cada una de las
    8 direcciones -> vector de longitud fija (8), invariante a la
    posición del contorno y aproximadamente robusto a la escala.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.zeros(config.FREEMAN_N_DIRECTIONS, dtype="float32")

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 2:
        return np.zeros(config.FREEMAN_N_DIRECTIONS, dtype="float32")

    diffs = np.diff(contour, axis=0, append=contour[:1])
    diffs = np.clip(diffs, -1, 1)  # cuantiza pasos a {-1,0,1} (8-conectividad)

    chain_hist = np.zeros(config.FREEMAN_N_DIRECTIONS, dtype="float64")
    for step in diffs:
        if step[0] == 0 and step[1] == 0:
            continue
        # distancia euclidiana a cada una de las 8 direcciones candidatas
        dists = np.sum((_FREEMAN_DIRS - step) ** 2, axis=1)
        direction = int(np.argmin(dists))
        chain_hist[direction] += 1

    chain_hist /= (chain_hist.sum() + 1e-7)
    return chain_hist.astype("float32")


# ---------------------------------------------------------------------------
# Extracción combinada (pipeline completo para una imagen)
# ---------------------------------------------------------------------------
def extract_features(image_path: str) -> Dict[str, np.ndarray]:
    """
    Ejecuta el pipeline completo sobre una imagen y devuelve un
    diccionario con cada descriptor por separado además del vector
    combinado. Mantener los descriptores separados facilita el
    análisis de "qué descriptor aportó más valor" pedido en la
    sección de Conclusiones del reporte.
    """
    img_bgr = _load_and_resize(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = _binary_mask(gray)

    lbp_feat = extract_lbp(gray)
    hog_feat = extract_hog(gray)
    hu_feat = extract_hu_moments(mask)
    freeman_feat = extract_freeman(mask)

    combined = np.concatenate([lbp_feat, hog_feat, hu_feat, freeman_feat])

    return {
        "lbp": lbp_feat,
        "hog": hog_feat,
        "hu": hu_feat,
        "freeman": freeman_feat,
        "combined": combined,
    }


if __name__ == "__main__":
    # Prueba rápida sobre una imagen dummy generada en memoria
    dummy = np.zeros((128, 128, 3), dtype="uint8")
    cv2.circle(dummy, (64, 64), 40, (255, 255, 255), -1)
    cv2.imwrite("/tmp/_dummy_ball.jpg", dummy)
    feats = extract_features("/tmp/_dummy_ball.jpg")
    for k, v in feats.items():
        print(f"{k}: shape={v.shape}")
