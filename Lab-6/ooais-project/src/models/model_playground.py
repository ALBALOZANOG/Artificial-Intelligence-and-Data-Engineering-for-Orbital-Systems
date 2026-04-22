import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Task 1: Create the Script and Verify Input Files ---
def validate_input_files():
    files = [
        "data/processed/model_features.csv",
        "data/processed/model_labels.csv"
    ]
    missing = [f for f in files if not Path(f).exists()]
    
    if missing:
        print("Error: missing required input file(s):")
        for m in missing:
            print(f"- {m}")
        exit(1)

# --- Task 2: Loading Input Data ---
def load_data():
    print("\n=== Model Playground: Loading Data ===")
    features_path = "data/processed/model_features.csv"
    labels_path = "data/processed/model_labels.csv"
    print(f"Feature file: {features_path}")
    print(f"Label file: {labels_path}")
    
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    return features_df, labels_df

# --- Task 3: Inspecting and Validating Loaded Data ---
def inspect_data(features_df, labels_df):
    print("\n=== Model Playground: Data Inspection ===")
    if features_df.empty or labels_df.empty:
        print("Error: Dataset is empty.")
        exit(1)
    if len(features_df) != len(labels_df):
        print("Error: Row count mismatch between features and labels.")
        exit(1)
    if "anomaly_flag" not in labels_df.columns:
        print("Error: 'anomaly_flag' column missing.")
        exit(1)
        
    print(f"Number of samples: {len(features_df)}")
    print(f"Number of features: {features_df.shape[1]}")
    print(f"Feature columns: {list(features_df.columns)}")
    print(f"Target values detected: {labels_df['anomaly_flag'].unique().tolist()}")

# --- Task 4: Preparing Machine Learning Inputs (X and y) ---
def prepare_features_and_labels(features_df, labels_df):
    print("\n=== Model Playground: Preparing Features and Labels ===")
    X = features_df.values
    y = labels_df["anomaly_flag"].astype(int).values
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y

# --- Task 5: Splitting Data into Training and Test Sets ---
def split_data(X, y):
    print("\n=== Model Playground: Train/Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# --- Task 6: Defining Models for Comparison ---
def define_models():
    models = {
        "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    return models

# --- Task 7: Training All Models ---
def train_models(models, X_train, y_train):
    print("\n=== Model Playground: Training Models ===")
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name}: trained")
        trained_models[name] = model
    return trained_models

# --- Task 8: Generating Predictions ---
def generate_predictions(trained_models, X_test):
    results = []
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        results.append({"name": name, "model": model, "y_pred": y_pred})
    return results

# --- Task 9: Inspecting Example Predictions ---
def print_example_predictions(prediction_results, y_test, num_examples=5):
    print("\n=== Model Playground: Example Predictions ===")
    for i in range(num_examples):
        line = f"True: {y_test[i]}"
        for res in prediction_results:
            line += f" | {res['name']}: {res['y_pred'][i]}"
        print(line)

# --- Task 10: Comparing Models Using Accuracy ---
def compute_accuracy(prediction_results, y_test):
    print("\n=== Model Playground: Accuracy Comparison ===")
    for res in prediction_results:
        acc = accuracy_score(y_test, res["y_pred"])
        res["accuracy"] = acc
        print(f"{res['name']}: {acc:.4f}")
    return prediction_results

# --- Task 11: Computing Detailed Evaluation Metrics ---
def compute_detailed_metrics(prediction_results, y_test):
    print("\n=== Model Playground: Detailed Evaluation ===")
    for res in prediction_results:
        y_pred = res["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        res["confusion_matrix"] = cm
        res["classification_report"] = report
        
        print(f"\nModel: {res['name']}")
        print(f"Accuracy: {res['accuracy']:.4f}")
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:")
        print("-" * 60)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
        print("-" * 60)
        for label in ['0', '1']:
            lbl_name = "0 (normal)" if label == '0' else "1 (anomaly)"
            m = report[label]
            print(f"{lbl_name:<15} {m['precision']:<10.2f} {m['recall']:<10.2f} {m['f1-score']:<10.2f} {int(m['support']):<10}")
        print("-" * 60)
    return prediction_results

# --- Task 12: Ranking Models by Performance ---
def rank_models(evaluation_results):
    print("\n=== Model Playground: Ranking ===")
    sorted_results = sorted(evaluation_results, key=lambda x: x["accuracy"], reverse=True)
    for i, res in enumerate(sorted_results, 1):
        print(f"{i}. {res['name']} - {res['accuracy']:.4f}")
    return sorted_results

# --- Task 13: Running Controlled Experiments ---
def run_experiments(X_train, X_test, y_train, y_test):
    print("\n=== Model Playground: Controlled Experiments ===")
    exp_results = []
    
    # Experiment 1: Decision Tree Depth
    depths = [2, 3, 5]
    for d in depths:
        model = DecisionTreeClassifier(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Decision Tree (max_depth={d}): {acc:.4f}")
        exp_results.append({"name": f"Decision Tree (max_depth={d})", "accuracy": acc, "param": d, "type": "DT"})

    # Experiment 2: Random Forest Size
    sizes = [5, 10, 50]
    for s in sizes:
        model = RandomForestClassifier(n_estimators=s, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Random Forest (n_estimators={s}): {acc:.4f}")
        exp_results.append({"name": f"Random Forest (n_estimators={s})", "accuracy": acc, "param": s, "type": "RF"})
    
    return exp_results

# --- Task 21: Optional Challenge (Disagreements) ---
def analyze_disagreements(prediction_results, y_test):
    print("\n=== Model Playground: Analyzing Disagreements ===")
    # Identificamos desacuerdos entre los 3 modelos base
    preds = [res["y_pred"] for res in prediction_results]
    disagreement_indices = []
    for i in range(len(y_test)):
        unique_preds = set(p[i] for p in preds)
        if len(unique_preds) > 1:
            disagreement_indices.append(i)
    
    print(f"Found {len(disagreement_indices)} cases where models disagree.")
    return disagreement_indices

# --- Task 14 & 15: Saving Summary and Visualizations ---
def save_experiment_summary(features_path, labels_path, X, X_train, X_test, ranked_models, experiment_results, disagreements):
    print("\n=== Model Playground: Saving Summary ===")
    report_path = "reports/model_playground_summary.txt"
    best_model = ranked_models[0]
    
    with open(report_path, "w") as f:
        f.write("OOAIS Model Playground Summary\n")
        f.write("=============================\n\n")
        
        f.write("Dataset information\n------------------\n")
        f.write(f"Features: {features_path}\nLabels: {labels_path}\n\n")
        
        f.write("Dataset statistics\n------------------\n")
        f.write(f"Number of samples: {X.shape[0]}\n")
        f.write(f"Number of features: {X.shape[1]}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n\n")
        
        f.write("Model comparison results\n------------------------\n")
        for res in ranked_models:
            f.write(f"- {res['name']}: {res['accuracy']:.4f}\n")
        
        f.write(f"\nBest-performing model\n---------------------\n")
        f.write(f"{best_model['name']} achieved the highest accuracy: {best_model['accuracy']:.4f}\n\n")
        
        f.write("Selected evaluation metrics (Anomaly Class)\n------------------------------------------\n")
        best_rep = best_model['classification_report']['1']
        f.write(f"Precision: {best_rep['precision']:.2f}\n")
        f.write(f"Recall: {best_rep['recall']:.2f}\n")
        f.write(f"F1-score: {best_rep['f1-score']:.2f}\n\n")
        
        f.write("Results of controlled experiments\n---------------------------------\n")
        for exp in experiment_results:
            f.write(f"- {exp['name']}: {exp['accuracy']:.4f}\n")
            
        f.write("\nInterpretation and Reflection (Task 15)\n---------------------------------------\n")
        f.write(f"1. Highest Accuracy: {best_model['name']}.\n")
        f.write("2. Differences: Small (all models > 95%), but tree-based models were more consistent.\n")
        f.write("3. Complexity: Not always; a simple Decision Tree matched the Random Forest.\n")
        f.write("4. Promising Model: Random Forest, due to its ensemble stability.\n")
        f.write("5. Reliable Anomaly Detection: Tree-based models (Precision 1.00, Recall 0.93).\n")
        f.write("6. Accuracy Limits: It hides the poor recall of Logistic Regression (0.71) for rare anomalies.\n\n")

        f.write("Optional Challenge: Disagreements (Task 21)\n------------------------------------------\n")
        f.write(f"Total disagreements: {len(disagreements)}\n")
        f.write("- Models disagree on 'borderline' cases that lack clear linear separation.\n")
        f.write("- Tree-based models are correct more often in these cases.\n")
        f.write("- These cases are ambiguous as they likely reside near decision boundaries.\n\n")

        f.write("Final Conclusion\n----------------\n")
        f.write(f"Tree-based models are the best candidates for this task due to superior anomaly detection recall.")

    print(f"Saved file: {report_path}")

def create_metric_plots(ranked_models):
    print("\n=== Model Playground: Saving Visualizations ===")
    names = [r["name"] for r in ranked_models]
    accs = [r["accuracy"] for r in ranked_models]
    pre = [r["classification_report"]["1"]["precision"] for r in ranked_models]
    rec = [r["classification_report"]["1"]["recall"] for r in ranked_models]
    f1s = [r["classification_report"]["1"]["f1-score"] for r in ranked_models]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [accs, pre, rec, f1s]
    titles = ["Accuracy", "Precision (Anomaly)", "Recall (Anomaly)", "F1-score (Anomaly)"]
    
    for ax, data, title in zip(axes.flat, metrics, titles):
        bars = ax.bar(names, data, color='skyblue', edgecolor='navy')
        ax.set_title(title, fontweight='bold')
        
        min_val = min(data)
        ax.set_ylim(max(0, min_val - 0.05), 1.02) 
        
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("reports/model_comparison_panel.png")
    print("Saved file: reports/model_comparison_panel.png")
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    validate_input_files()
    f_df, l_df = load_data()
    inspect_data(f_df, l_df)
    X, y = prepare_features_and_labels(f_df, l_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model_defs = define_models()
    trained = train_models(model_defs, X_train, y_train)
    
    results = generate_predictions(trained, X_test)
    print_example_predictions(results, y_test)
    
    results = compute_accuracy(results, y_test)
    results = compute_detailed_metrics(results, y_test)
    
    ranked = rank_models(results)
    disagreements = analyze_disagreements(results, y_test)
    exp_results = run_experiments(X_train, X_test, y_train, y_test)
    
    save_experiment_summary(
        "data/processed/model_features.csv",
        "data/processed/model_labels.csv",
        X, X_train, X_test, ranked, exp_results, disagreements
    )
    create_metric_plots(ranked)