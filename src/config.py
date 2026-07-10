"""
config.py
---------
Configuración centralizada del proyecto de clasificación de pelotas deportivas.

Concentrar aquí todas las rutas e hiperparámetros permite modificar el
comportamiento del pipeline completo (extracción -> almacenamiento ->
entrenamiento -> evaluación) sin tocar la lógica de negocio en cada módulo.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# DATA_ROOT debe apuntar a la carpeta que contiene "train/" y "test/",
# cada una con subcarpetas por clase (ver estructura del dataset de Kaggle:
# "Sports Balls - Multiclass Image Classification").
DATA_ROOT = Path("./data")          # <-- AJUSTAR a la ruta real del dataset
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

RESULTS_DIR = Path("./results")
FEATURES_DIR = RESULTS_DIR / "features"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

FEATURES_JSON_TRAIN = FEATURES_DIR / "features_train.json"
FEATURES_JSON_TEST = FEATURES_DIR / "features_test.json"

# ---------------------------------------------------------------------------
# Clases del dataset (se detectan automáticamente, pero se listan aquí
# como referencia y para fijar un orden determinista de las etiquetas)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "american_football", "baseball", "basketball", "billiard_ball",
    "bowling_ball", "cricket_ball", "football", "golf_ball",
    "hockey_ball", "hockey_puck", "rugby_ball", "shuttlecock",
    "table_tennis_ball", "tennis_ball", "volleyball",
]

# ---------------------------------------------------------------------------
# Hiperparámetros de extracción de características
# ---------------------------------------------------------------------------
IMG_SIZE = (128, 128)          # tamaño fijo para descriptores clásicos (HOG/LBP)
CNN_IMG_SIZE = (64, 64)        # tamaño de entrada para la CNN (más liviano)

LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

FREEMAN_N_DIRECTIONS = 8       # 8-conectividad estándar del código de Freeman

# Reducción de dimensionalidad para los modelos clásicos (el vector combinado
# LBP+HOG+Hu+Freeman puede superar las 1800 dimensiones; con pocas muestras
# por clase esto degrada KNN/SVM/NB por la maldición de la dimensionalidad).
PCA_COMPONENTS = 100

# ---------------------------------------------------------------------------
# Hiperparámetros de entrenamiento
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.15   # tomado del propio train set, para early-stopping/CNN

KNN_N_NEIGHBORS = 7
SVM_KERNEL = "rbf"
SVM_C = 10.0
SVM_GAMMA = "scale"

CNN_EPOCHS = 15
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 1e-3

# ---------------------------------------------------------------------------
# Utilidad: crear carpetas de resultados si no existen
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in (RESULTS_DIR, FEATURES_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
