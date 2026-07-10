# Clasificación de Pelotas Deportivas — Descriptores Clásicos + ML/DL

Sistema de clasificación de imágenes para 15 tipos de pelotas deportivas
(fútbol americano, béisbol, baloncesto, billar, boliche, críquet, fútbol,
golf, hockey césped, hockey puck, rugby, bádminton, tenis de mesa, tenis
y voleibol), a partir de descriptores geométricos y de región clásicos:

- **LBP** (Local Binary Patterns) — textura local
- **HOG** (Histogram of Oriented Gradients) — gradientes y forma
- **Momentos Invariantes de Hu** — forma global invariante a escala/rotación
- **Código de Cadena de Freeman** — descriptor de contorno

Estos descriptores se combinan en un único vector de características y
se comparan cuatro clasificadores sobre él: **KNN**, **SVM**, **Naive
Bayes** y una **CNN** entrenada end-to-end sobre las imágenes crudas
(PyTorch), como contraste entre ingeniería de características manual y
representación aprendida.

Dataset: [Sports balls – multiclass image classification](https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification) (Cortinhas, 2022).

## Resultados

Sobre el conjunto de prueba (1,841 imágenes, no vistas durante el
entrenamiento):

| Modelo | Accuracy | F1 macro | F1 ponderado |
|---|---|---|---|
| **SVM** | **0.539** | 0.535 | 0.539 |
| CNN | 0.394 | 0.380 | 0.381 |
| KNN | 0.342 | 0.338 | 0.341 |
| Naive Bayes | 0.276 | 0.266 | 0.266 |

Matrices de confusión, gráfica comparativa y CSVs completos se generan
en `results/figures/` al correr el pipeline.

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Preparar el dataset

Descarga el dataset desde Kaggle y colócalo con esta estructura:

```
data/
├── train/
│   ├── american_football/*.jpg
│   ├── baseball/*.jpg
│   └── ... (15 clases)
└── test/
    ├── american_football/*.jpg
    └── ...
```

## Uso

```bash
python main.py --data_root ./data
```

Esto ejecuta el pipeline completo:

1. Carga `train/` y `test/` (`src/data_loader.py`).
2. Extrae LBP + HOG + Hu + Freeman por imagen y las guarda en
   `results/features/features_train.json` y `features_test.json`
   (`src/feature_extraction.py`, `src/storage.py`).
3. Entrena KNN, SVM y Naive Bayes sobre el vector combinado
   (`src/classical_models.py`).
4. Entrena una CNN sobre las imágenes crudas (`src/cnn_model.py`).
5. Evalúa los 4 modelos y genera matrices de confusión y tabla
   comparativa (`src/evaluate.py`).

### Flags opcionales

```bash
# Reutilizar características ya extraídas
python main.py --data_root ./data --skip_extraction

# Omitir el entrenamiento de la CNN
python main.py --data_root ./data --skip_cnn
```

## Estructura del proyecto

```
sports_ball_classifier/
├── main.py                     # orquestador del pipeline
├── requirements.txt
├── data/                       # dataset (no versionado)
├── results/                    # salidas generadas (no versionado)
└── src/
    ├── config.py                # rutas e hiperparámetros centralizados
    ├── data_loader.py            # lectura de train/ y test/
    ├── feature_extraction.py     # LBP, HOG, Hu, Freeman
    ├── storage.py                # serialización de características (JSON)
    ├── classical_models.py       # pipelines de KNN, SVM, Naive Bayes
    ├── cnn_model.py               # arquitectura y entrenamiento de la CNN
    └── evaluate.py                # métricas, matrices de confusión, comparativa
```

## Salidas generadas

```
results/
├── features/
│   ├── features_train.json
│   └── features_test.json
└── figures/
    ├── confusion_matrix_KNN.png / .csv
    ├── confusion_matrix_SVM.png / .csv
    ├── confusion_matrix_NaiveBayes.png / .csv
    ├── confusion_matrix_CNN.png / .csv
    ├── model_comparison.png
    └── model_comparison.csv
```

## Stack técnico

Python · OpenCV · scikit-image · scikit-learn · PyTorch · pandas · matplotlib

## Autor

Johan — Ingeniería en Inteligencia Artificial, Universidad de Xalapa.
Proyecto desarrollado para la asignatura de Procesamiento de Imágenes.
