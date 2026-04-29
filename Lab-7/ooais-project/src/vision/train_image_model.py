import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from src.vision.feature_extractor import extract_features

DATASET_DIR = Path("data/processed/images")
MODELS_DIR = Path("models")

def load_image_split(split_dir):
    X = []
    y = []
    class_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted([
            path for path in class_dir.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        for image_path in image_files:
            with Image.open(image_path) as image:
                features = extract_features(image)
                X.append(features)
                y.append(class_name)
    return np.array(X), np.array(y)

def load_training_and_test_data():
    X_train, y_train = load_image_split(DATASET_DIR / "train")
    X_test, y_test = load_image_split(DATASET_DIR / "test")
    print("=== Image ML Dataset ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }
    
    results = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Training and Evaluation ===")
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            "model_name": name,
            "accuracy": accuracy,
            "training_time": training_time,
            "model": model
        })
        
        joblib.dump(model, MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib")
        
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.4f} s\n")
        
    return results

def plot_accuracy_vs_time(results):
    names = [r["model_name"] for r in results]
    accs = [r["accuracy"] for r in results]
    times = [r["training_time"] for r in results]
    
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(names):
        plt.scatter(times[i], accs[i], s=100, label=name)
        plt.annotate(name, (times[i], accs[i]), xytext=(5, 5), textcoords='offset points')
        
    plt.xlabel("Training Time (s)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Training Time")
    plt.grid(True)
    plt.savefig("model_comparison.png")
    print("Comparison plot saved as model_comparison.png")

def main():
    X_train, X_test, y_train, y_test = load_training_and_test_data()
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    plot_accuracy_vs_time(results)

if __name__ == "__main__":
    main()