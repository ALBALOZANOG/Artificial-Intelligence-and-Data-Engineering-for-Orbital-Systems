import csv
from datetime import datetime

INPUT_FILE = "data/processed/observations_valid.csv"
FEATURES_OUTPUT = "data/processed/model_features.csv"
LABELS_OUTPUT = "data/processed/model_labels.csv"

REQUIRED_COLUMNS = ["temperature", "velocity", "altitude", "signal_strength"]


def is_valid_record(row):
    try:
        temperature = float(row["temperature"])
        velocity = float(row["velocity"])
        altitude = float(row["altitude"])
        signal_strength = float(row["signal_strength"])

        if altitude < 0:
            return False

        return True

    except (ValueError, KeyError):
        return False


def main():
    print("=== ML Input Preparation: Loading and Conversion ===")
    print(f"Input file: {INPUT_FILE}")

    total_records = 0
    accepted_records = 0
    rejected_records = 0

    clean_data = []
    labels = []

    # LOAD + VALIDATION
    with open(INPUT_FILE, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            total_records += 1

            if any(row[col] == "" or row[col] is None for col in REQUIRED_COLUMNS):
                rejected_records += 1
                continue

            if is_valid_record(row):
                clean_row = row.copy()

                for col in REQUIRED_COLUMNS:
                    clean_row[col] = float(row[col])

                clean_data.append(clean_row)

                # Guardar label en el mismo orden
                labels.append({"anomaly_flag": row["anomaly_flag"]})

                accepted_records += 1
            else:
                rejected_records += 1

    print(f"Records loaded: {total_records}")
    print(f"Records accepted: {accepted_records}")
    print(f"Records rejected: {rejected_records}")

    if not clean_data:
        print("No valid data available for normalization.")
        return

    # NORMALIZATION
    print("\n=== ML Input Preparation: Normalization ===")

    min_vals = {}
    max_vals = {}

    for col in REQUIRED_COLUMNS:
        values = [row[col] for row in clean_data]
        min_vals[col] = min(values)
        max_vals[col] = max(values)

    normalized_data = []

    for row in clean_data:
        normalized_row = row.copy()

        for col in REQUIRED_COLUMNS:
            min_val = min_vals[col]
            max_val = max_vals[col]

            if max_val == min_val:
                normalized_value = 0.0
            else:
                normalized_value = (row[col] - min_val) / (max_val - min_val)

            normalized_row[col] = normalized_value

        normalized_data.append(normalized_row)

    all_in_range = True

    for row in normalized_data:
        for col in REQUIRED_COLUMNS:
            if not (0.0 <= row[col] <= 1.0):
                all_in_range = False
                break

    if all_in_range:
        print("Normalization completed successfully.")
        print("All selected numerical features are in range [0,1].")
    else:
        print("Warning: Some values are out of range!")

    # DERIVED FEATURES
    print("\n=== ML Input Preparation: Derived Features ===")

    feature_1 = "temperature_velocity_interaction"
    feature_2 = "altitude_signal_ratio"

    for row in normalized_data:
        if feature_1 not in row:
            row[feature_1] = row["temperature"] * row["velocity"]

        if feature_2 not in row:
            row[feature_2] = row["altitude"] / (row["signal_strength"] + 0.0001)

    print("New features added:")
    print(f"- {feature_1}")
    print(f"- {feature_2}")

    # TEMPORAL FEATURES
    print("\n=== ML Input Preparation: Temporal Features ===")

    temporal_feature = "hour_normalized"

    for row in normalized_data:
        if temporal_feature not in row:
            try:
                timestamp_str = row["timestamp"]
                dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                hour = dt.hour
                row[temporal_feature] = hour / 24
            except Exception:
                row[temporal_feature] = 0.0

    print("New feature added:")
    print(f"- {temporal_feature}")

    # FEATURE SELECTION
    print("\n=== ML Input Preparation: Feature Selection ===")

    selected_features = [
        "temperature",
        "velocity",
        "altitude",
        "signal_strength",
        "temperature_velocity_interaction",
        "altitude_signal_ratio",
        "hour_normalized"
    ]

    final_data = []

    for row in normalized_data:
        selected_row = {col: row[col] for col in selected_features}
        final_data.append(selected_row)

    print("Selected features:")
    for feature in selected_features:
        print(f"- {feature}")

    # SAVE OUTPUTS
    print("\n=== ML Input Preparation: Saving Outputs ===")

    # Guardar features
    with open(FEATURES_OUTPUT, mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=selected_features)
        writer.writeheader()
        writer.writerows(final_data)

    # Guardar labels
    with open(LABELS_OUTPUT, mode="w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["anomaly_flag"])
        writer.writeheader()
        writer.writerows(labels)

    print(f"Saved file: {FEATURES_OUTPUT}")
    print(f"Saved file: {LABELS_OUTPUT}")
    print(f"Number of records: {len(final_data)}")
    print(f"Number of features: {len(selected_features)}")

    if labels:
        print("\nExample label record:")
        print(labels[0])


if __name__ == "__main__":
    main()