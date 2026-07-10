"""
cnn_model.py
------------
Arquitectura y rutina de entrenamiento de una CNN convolucional simple,
implementada en PyTorch, para comparar Deep Learning contra los
descriptores clásicos + ML (KNN/SVM/Naive Bayes).

A diferencia de los modelos clásicos, la CNN NO recibe los vectores
LBP/HOG/Hu/Freeman como entrada: aprende directamente sus propios
filtros convolucionales a partir de los píxeles crudos de la imagen
(redimensionada a config.CNN_IMG_SIZE). Esta es la comparación central
que pide la sección de Resultados de la rúbrica: "ML clásico basado en
descriptores hechos a mano" vs. "Deep Learning con representaciones
aprendidas".
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config


# ---------------------------------------------------------------------------
# Dataset de PyTorch que lee imágenes crudas desde disco
# ---------------------------------------------------------------------------
class BallImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str]], label_to_idx: dict):
        self.samples = samples
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, config.CNN_IMG_SIZE, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        tensor_img = torch.from_numpy(img)
        tensor_label = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return tensor_img, tensor_label


# ---------------------------------------------------------------------------
# Arquitectura CNN
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    CNN compacta de 3 bloques convolucionales + cabeza densa.
    Diseñada para converger rápido en datasets de tamaño moderado
    (miles de imágenes, no millones) sin sobreajustar de inmediato.
    """

    def __init__(self, n_classes: int, img_size: Tuple[int, int]):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # img_size / 8
        )
        reduced_h = img_size[0] // 8
        reduced_w = img_size[1] // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced_h * reduced_w, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Rutina de entrenamiento
# ---------------------------------------------------------------------------
def train_cnn(train_samples, val_samples, label_to_idx: dict, verbose: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BallImageDataset(train_samples, label_to_idx)
    val_ds = BallImageDataset(val_samples, label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=config.CNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.CNN_BATCH_SIZE, shuffle=False)

    model = SimpleCNN(n_classes=len(label_to_idx), img_size=config.CNN_IMG_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.CNN_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, config.CNN_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_acc = _evaluate_loader(model, val_loader, device)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"[CNN] Epoch {epoch:02d}/{config.CNN_EPOCHS} "
                  f"- train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, device


def _evaluate_loader(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return correct / max(total, 1)


def predict_cnn(model, device, samples, label_to_idx: dict):
    """Devuelve (y_true, y_pred) como arrays de strings de clase."""
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    ds = BallImageDataset(samples, label_to_idx)
    loader = DataLoader(ds, batch_size=config.CNN_BATCH_SIZE, shuffle=False)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend([idx_to_label[p] for p in preds])
            y_true.extend([idx_to_label[l] for l in labels.numpy()])
    return np.array(y_true), np.array(y_pred)
