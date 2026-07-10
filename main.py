"""
main.py
-------
Punto de entrada único del proyecto. Orquesta el pipeline completo:

    1. Carga del dataset (train/ y test/)
    2. Extracción de descriptores (LBP, HOG, Hu, Freeman) -> JSON
    3. Entrenamiento de KNN, SVM y Naive Bayes sobre el vector combinado
    4. Entrenamiento de una CNN sobre las imágenes crudas
    5. Evaluación de los 4 modelos: accuracy + matriz de confusión
    6. Tabla comparativa final

Uso:
    python main.py --data_root /ruta/al/dataset
    python main.py --data_root /ruta/al/dataset --skip_extraction
    python main.py --data_root /ruta/al/dataset --skip_cnn

El flag --skip_extraction permite reutilizar un features_*.json ya
generado (evita recalcular los descriptores en cada corrida).
El flag --skip_cnn permite iterar rápido sobre los modelos clásicos
sin pagar el costo de entrenar la red neuronal.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
import data_loader
import feature_extraction
import storage
import classical_models
import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de clasificación de pelotas deportivas")
    parser.add_argument("--data_root", type=str, default=str(config.DATA_ROOT),
                         help="Carpeta que contiene train/ y test/")
    parser.add_argument("--skip_extraction", action="store_true",
                         help="Reutiliza los features_*.json ya generados en results/features/")
    parser.add_argument("--skip_cnn", action="store_true",
                         help="Omite el entrenamiento de la CNN (solo modelos clásicos)")
    return parser.parse_args()


def extract_and_store(samples, out_json: Path, split_name: str):
    print(f"\n[main] Extrayendo características ({split_name}): {len(samples)} imágenes...")
    records = []
    t0 = time.time()
    for i, (path, label) in enumerate(samples, 1):
        feats = feature_extraction.extract_features(path)
        records.append({"image_path": path, "label": label, "features": feats})
        if i % 200 == 0 or i == len(samples):
            elapsed = time.time() - t0
            print(f"  [{split_name}] {i}/{len(samples)} imágenes procesadas ({elapsed:.1f}s)")
    storage.save_features_json(records, out_json)
    return records


def main():
    args = parse_args()
    config.DATA_ROOT = Path(args.data_root)
    config.TRAIN_DIR = config.DATA_ROOT / "train"
    config.TEST_DIR = config.DATA_ROOT / "test"
    config.ensure_dirs()

    # -----------------------------------------------------------------
    # 1. Carga del dataset
    # -----------------------------------------------------------------
    train_samples = data_loader.list_images(config.TRAIN_DIR)
    test_samples = data_loader.list_images(config.TEST_DIR)
    class_names = sorted(set(label for _, label in train_samples))

    print(f"[main] Clases detectadas ({len(class_names)}): {class_names}")
    print(f"[main] Train: {len(train_samples)} imágenes | Test: {len(test_samples)} imágenes")

    # -----------------------------------------------------------------
    # 2. Extracción de características (o carga desde JSON existente)
    # -----------------------------------------------------------------
    if args.skip_extraction and config.FEATURES_JSON_TRAIN.exists():
        print("[main] Cargando características ya extraídas desde JSON...")
        train_records = storage.load_features_json(config.FEATURES_JSON_TRAIN)
        test_records = storage.load_features_json(config.FEATURES_JSON_TEST)
    else:
        train_records = extract_and_store(train_samples, config.FEATURES_JSON_TRAIN, "train")
        test_records = extract_and_store(test_samples, config.FEATURES_JSON_TEST, "test")

    X_train, y_train = storage.to_matrix(train_records, "combined")
    X_test, y_test = storage.to_matrix(test_records, "combined")
    print(f"[main] Dimensión del vector combinado: {X_train.shape[1]}")

    # -----------------------------------------------------------------
    # 3. Entrenamiento y evaluación de modelos clásicos
    # -----------------------------------------------------------------
    results = []
    trained_models = {}
    models = classical_models.get_classical_models()

    for name, pipeline in models.items():
        print(f"\n[main] Entrenando {name}...")
        t0 = time.time()
        pipeline = classical_models.train_model(pipeline, X_train, y_train)
        print(f"[main] {name} entrenado en {time.time() - t0:.1f}s")
        trained_models[name] = pipeline

        y_pred = pipeline.predict(X_test)
        results.append(evaluate.evaluate_predictions(name, y_test, y_pred, class_names))

    # -----------------------------------------------------------------
    # 4. Entrenamiento y evaluación de la CNN (imágenes crudas)
    # -----------------------------------------------------------------
    if not args.skip_cnn:
        import cnn_model
        from sklearn.model_selection import train_test_split

        label_to_idx = {c: i for i, c in enumerate(class_names)}
        cnn_train, cnn_val = train_test_split(
            train_samples, test_size=config.VALIDATION_SPLIT,
            random_state=config.RANDOM_STATE,
            stratify=[l for _, l in train_samples],
        )

        print("\n[main] Entrenando CNN (PyTorch)...")
        t0 = time.time()
        model, device = cnn_model.train_cnn(cnn_train, cnn_val, label_to_idx)
        print(f"[main] CNN entrenada en {time.time() - t0:.1f}s")

        y_true_cnn, y_pred_cnn = cnn_model.predict_cnn(model, device, test_samples, label_to_idx)
        results.append(evaluate.evaluate_predictions("CNN", y_true_cnn, y_pred_cnn, class_names))
    else:
        print("\n[main] --skip_cnn activo: se omite el entrenamiento de la red neuronal.")

    # -----------------------------------------------------------------
    # 5. Tabla comparativa final
    # -----------------------------------------------------------------
    evaluate.build_comparison_table(results)
    print(f"\n[main] Listo. Resultados guardados en: {config.FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
