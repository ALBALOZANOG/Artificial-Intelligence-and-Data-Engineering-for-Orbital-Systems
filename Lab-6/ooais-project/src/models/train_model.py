import os
import csv
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

FEATURES_PATH = "data/processed/model_features.csv"
LABELS_PATH = "data/processed/model_labels.csv"
MODEL_PATH = "results/decision_tree_model.joblib"
EVAL_PATH = "results/model_evaluation.txt"
REPORT_PATH = "reports/model_training_summary.txt"


def load_csv(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)

    return header, data


def main():
    print("=== Machine Learning: Loading Feature Dataset ===")

    # Load features
    print(f"Input file: {FEATURES_PATH}")
    feature_header, feature_rows = load_csv(FEATURES_PATH)

    print(f"Records loaded: {len(feature_rows)}")
    print(f"Columns: {feature_header}")

    # Load labels
    label_header, label_rows = load_csv(LABELS_PATH)

    if "anomaly_flag" not in label_header:
        raise ValueError("Column 'anomaly_flag' not found")

    target_idx = label_header.index("anomaly_flag")

    # Prepare X and y
    print("\n=== Machine Learning: Preparing Features and Target ===")

    X = []
    y = []

    for i in range(len(feature_rows)):
        X.append([float(v) for v in feature_rows[i]])
        y.append(int(label_rows[i][target_idx]))

    print(f"Number of samples in X: {len(X)}")
    print(f"Number of labels in y: {len(y)}")
    print(f"Target values detected: {sorted(list(set(y)))}")

    # Split
    print("\n=== Machine Learning: Train/Test Split ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train
    print("\n=== Machine Learning: Model Training ===")

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    print("Model: DecisionTreeClassifier")
    print("Training completed successfully.")

    # Predict
    print("\n=== Machine Learning: Prediction ===")

    predictions = model.predict(X_test)

    print("Predictions generated for test set.")
    print(f"Number of predictions: {len(predictions)}")
    print("Example predictions:")
    print(predictions[:5])

    # Evaluate
    print("\n=== Machine Learning: Evaluation ===")

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Ensure folders exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Save model + inspect
    print("\n=== Machine Learning: Saving and Inspecting Model ===")

    joblib.dump(model, MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Model type: {type(model).__name__}")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")

    print("Decision Tree Rules:")
    tree_rules = export_text(model, feature_names=feature_header)
    print(tree_rules)

    # Save evaluation
    print("\n=== Machine Learning: Saving Evaluation Results ===")

    with open(EVAL_PATH, "w") as f:
        f.write("OOAIS Model Evaluation\n")
        f.write("======================\n\n")
        f.write(f"Model: DecisionTreeClassifier\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")

    print(f"Saved file: {EVAL_PATH}")

    # Training report (Task 9)
    print("\n=== Machine Learning: Saving Training Report ===")

    with open(REPORT_PATH, "w") as f:
        f.write("OOAIS Model Training Summary\n")
        f.write("============================\n\n")

        f.write("Input datasets\n")
        f.write("--------------\n")
        f.write(f"{FEATURES_PATH}\n")
        f.write(f"{LABELS_PATH}\n\n")

        f.write("Dataset statistics\n")
        f.write("------------------\n")
        f.write(f"Number of samples: {len(X)}\n")
        f.write(f"Number of features: {len(feature_header)}\n\n")

        f.write("Model\n")
        f.write("-----\n")
        f.write("DecisionTreeClassifier\n\n")

        f.write("Train/Test split\n")
        f.write("----------------\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")

        f.write("Evaluation summary\n")
        f.write("------------------\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")

    print(f"Saved file: {REPORT_PATH}")


if __name__ == "__main__":
    main()
