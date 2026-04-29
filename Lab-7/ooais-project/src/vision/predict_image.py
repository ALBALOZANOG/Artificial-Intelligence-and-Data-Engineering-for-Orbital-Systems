from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from src.vision.feature_extractor import extract_features

MODELS_DIR = Path("models")

def load_all_models():
    model_files = {
        "Random Forest": "random_forest.joblib",
        "KNN": "knn.joblib",
        "Logistic Regression": "logistic_regression.joblib",
        "SVM": "svm.joblib"
    }
    
    loaded_models = {}
    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            loaded_models[name] = joblib.load(path)
        else:
            print(f"Warning: {name} not found at {path}")
            
    if not loaded_models:
        print("Error: No models found. Run train_image_model.py first.")
        raise SystemExit(1)
    
    print(f"Loaded {len(loaded_models)} models.")
    return loaded_models

def predict_with_all(models, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return
    
    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.copy()
        
        results_text = []
        print(f"=== Predictions for: {image_path} ===")
        
        for name, model in models.items():
            prediction = model.predict([features])[0]
            results_text.append(f"{name}: {prediction}")
            print(f"{name}: {prediction}")

        plt.figure(figsize=(12, 7))
        plt.imshow(image_for_plot)
        full_title = " | ".join(results_text)
        plt.title(full_title, fontsize=9)
        plt.axis("off")
        
        output_name = "multi_model_prediction.png"
        plt.savefig(output_name)
        print(f"\nVisualización guardada como: {output_name}")
        
        try:
            plt.show()
        except UserWarning:
            pass
        plt.close()

def main():
    models = load_all_models()
    
    image_path = "data/inference_samples/noise.jpg"
    
    predict_with_all(models, image_path)

if __name__ == "__main__":
    main()