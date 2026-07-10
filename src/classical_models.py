"""
classical_models.py
--------------------
Define y entrena los tres clasificadores clásicos exigidos por la
rúbrica: KNN, SVM y Naive Bayes, todos operando sobre los vectores de
características combinados (LBP + HOG + Hu + Freeman).

Cada modelo se envuelve en un Pipeline de scikit-learn:
    StandardScaler -> PCA -> Clasificador

Justificación de StandardScaler + PCA:
  - Los tres descriptores tienen escalas muy distintas (LBP y Freeman
    son histogramas normalizados en [0,1]; HOG puede tener cientos de
    componentes con distinta varianza; Hu-log tiene rango propio) por
    lo que estandarizar es indispensable antes de KNN/SVM (sensibles
    a la escala) y antes de PCA.
  - El vector combinado supera las 1800 dimensiones mientras el
    dataset tiene un número de muestras por clase mucho menor; PCA
    reduce a `config.PCA_COMPONENTS` componentes para mitigar la
    maldición de la dimensionalidad y acelerar el entrenamiento.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import config


def build_knn_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_STATE)),
        ("clf", KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)),
    ])


def build_svm_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_STATE)),
        ("clf", SVC(kernel=config.SVM_KERNEL, C=config.SVM_C, gamma=config.SVM_GAMMA,
                     probability=True, random_state=config.RANDOM_STATE)),
    ])


def build_naive_bayes_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=config.PCA_COMPONENTS, random_state=config.RANDOM_STATE)),
        ("clf", GaussianNB()),
    ])


def get_classical_models() -> dict:
    """Devuelve un diccionario {nombre_modelo: pipeline_sin_entrenar}."""
    return {
        "KNN": build_knn_pipeline(),
        "SVM": build_svm_pipeline(),
        "NaiveBayes": build_naive_bayes_pipeline(),
    }


def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline
