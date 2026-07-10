# Clasificación de Pelotas Deportivas — Descriptores Clásicos + ML/DL

Proyecto para el Examen Final: clasificación de 15 tipos de pelotas
deportivas usando descriptores geométricos/de región clásicos (LBP,
HOG, Momentos de Hu, Código de Freeman) y cuatro clasificadores
(KNN, SVM, Naive Bayes, CNN).

## 1. Instalación

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Preparar el dataset

Descarga el dataset "Sports Balls" y colócalo con esta estructura
(la misma del dataset de Kaggle, 80/20 train/test, subcarpeta por clase):

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

## 3. Ejecutar el pipeline completo

```bash
python main.py --data_root ./data
```

Esto hace, en orden:
1. Lee `train/` y `test/` (`src/data_loader.py`).
2. Extrae LBP + HOG + Hu + Freeman para cada imagen y guarda todo en
   `results/features/features_train.json` y `features_test.json`
   (`src/feature_extraction.py`, `src/storage.py`).
3. Entrena KNN, SVM y Naive Bayes sobre el vector combinado
   (`src/classical_models.py`).
4. Entrena una CNN (PyTorch) sobre las imágenes crudas
   (`src/cnn_model.py`).
5. Evalúa los 4 modelos: accuracy, matriz de confusión (PNG + CSV) y
   tabla comparativa final (`src/evaluate.py`), guardado todo en
   `results/figures/`.

### Flags útiles

```bash
# Reutilizar características ya extraídas (evita repetir el paso más lento)
python main.py --data_root ./data --skip_extraction

# Iterar rápido sobre los modelos clásicos sin entrenar la CNN
python main.py --data_root ./data --skip_cnn
```

## 4. Salidas generadas

```
results/
├── features/
│   ├── features_train.json     # estructura de datos (JSON) pedida en la rúbrica
│   └── features_test.json
└── figures/
    ├── confusion_matrix_KNN.png / .csv
    ├── confusion_matrix_SVM.png / .csv
    ├── confusion_matrix_NaiveBayes.png / .csv
    ├── confusion_matrix_CNN.png / .csv
    ├── model_comparison.png        # gráfica de barras comparativa
    └── model_comparison.csv        # tabla de accuracy/F1 por modelo
```

## 5. Mapeo del código a la rúbrica del examen

| Requisito de la rúbrica | Dónde está implementado |
|---|---|
| Pipeline de extracción de características | `src/feature_extraction.py` (LBP, HOG, Hu, Freeman) |
| Estructura de datos (JSON) | `src/storage.py` |
| Configuración de modelos, justificada | `src/classical_models.py`, `src/cnn_model.py` (docstrings explican cada decisión) |
| Tablas de confusión + Accuracy comparativo | `src/evaluate.py` |
| Código modularizado, comentado y organizado | separación en `data_loader / feature_extraction / storage / classical_models / cnn_model / evaluate`, cada uno con una sola responsabilidad |
| Resultados > azar matemático | con 15 clases, el azar es ~6.7% de accuracy; el pipeline fue validado end-to-end con un dataset sintético y todos los modelos superaron ese umbral |

## 6. Notas para el reporte técnico (Entregable 1)

- **Trabajos relacionados**: al describir LBP, HOG, Hu y Freeman en el
  estado del arte, cita los artículos originales (Ojala et al. 1996/2002
  para LBP; Dalal & Triggs 2005 para HOG; Hu 1962 para los momentos
  invariantes; Freeman 1961 para el código de cadena) y su aplicación a
  clasificación de objetos deportivos/esféricos.
- **Justificación JSON vs SQL**: está documentada como docstring en
  `src/storage.py`; cópiala/adáptala directamente a la sección de
  Materiales y Métodos.
- **Qué descriptor aportó más valor** (para la sección de Conclusiones):
  usa `storage.to_matrix(records, feature_key=...)` con `"lbp"`, `"hog"`,
  `"hu"` o `"freeman"` por separado, entrena el mismo modelo con cada
  uno, y compara accuracy — el código ya soporta esto sin modificaciones.
- **Clasificación ML clásico vs CNN**: la comparación directa es el
  contraste que pide la rúbrica en la Sección de Resultados de la
  presentación oral (5 min, núcleo de la defensa).
