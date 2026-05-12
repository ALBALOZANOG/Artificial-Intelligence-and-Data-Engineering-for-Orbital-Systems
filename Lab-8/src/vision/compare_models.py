import torch
import joblib
import time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from src.vision.train_cnn import create_dataloaders, get_device
from src.vision.cnn_model import SimpleCNN
from src.vision.feature_extractor import extract_features 

CNN_MODEL_PATH = Path("models/cnn_model.pt")
CLASSICAL_MODEL_PATH = Path("models/random_forest.joblib") 
REPORT_PATH = Path("reports/cnn_vs_ml.txt")

def evaluate_classical(model, test_dataset, class_names):
    all_preds = []
    all_labels = []
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    
    start_time = time.time()
    for image_path, label_idx in test_dataset.samples:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            features = extract_features(img)
            pred_string = model.predict(np.array(features).reshape(1, -1))[0]
            all_preds.append(name_to_idx[pred_string])
            all_labels.append(label_idx)
    
    end_time = time.time()
    acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return acc, end_time - start_time, conf_matrix

def evaluate_cnn(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    end_time = time.time()
    acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return acc, end_time - start_time, conf_matrix

def main():
    device = get_device()
    train_loader, test_loader, class_names = create_dataloaders()
    test_dataset = test_loader.dataset 

    cnn = SimpleCNN(num_classes=len(class_names)).to(device)
    if CNN_MODEL_PATH.exists():
        state_dict = torch.load(CNN_MODEL_PATH, map_location=device)
        cnn.load_state_dict(state_dict)
    else:
        return

    if CLASSICAL_MODEL_PATH.exists():
        ml_model = joblib.load(CLASSICAL_MODEL_PATH)
    else:
        return

    cnn_acc, cnn_time, cnn_cm = evaluate_cnn(cnn, test_loader, device)
    ml_acc, ml_time, ml_cm = evaluate_classical(ml_model, test_dataset, class_names)

    ml_errors = np.sum(ml_cm) - np.trace(ml_cm)
    cnn_errors = np.sum(cnn_cm) - np.trace(cnn_cm)

    report = f"""CNN VS CLASSICAL ML
===================
CLASSICAL ML:
Model: Random Forest
Training time: Fast (CPU-based)
Accuracy: {ml_acc:.4f}
CNN:
Model: Simple CNN
Training time: Slow (Epoch-based)
Accuracy: {cnn_acc:.4f}
BETTER ACCURACY:
{'Simple CNN' if cnn_acc > ml_acc else 'Random Forest'}
FASTER TRAINING:
Random Forest
FEWER CLASS CONFUSIONS:
{'Simple CNN' if cnn_errors < ml_errors else 'Random Forest'}
GENERALIZATION:
{'CNN captures spatial hierarchical patterns more effectively' if cnn_acc > ml_acc else 'Classical ML performs robustly with manual features'}
"""
    
    print(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()