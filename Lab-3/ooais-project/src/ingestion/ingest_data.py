# S2 imports
import json
import csv

#S3 paths
DATASET_PATH = "data/raw/orbital_observations.csv"
METADATA_PATH = "data/raw/metadata.json"
VALID_OUTPUT_PATH = "data/processed/observations_valid.csv"
INVALID_OUTPUT_PATH = "data/processed/observations_invalid.csv"
MODEL_INPUT_PATH = "data/processed/model_input.csv"
SUMMARY_PATH = "reports/ingestion_summary.txt"

#S4 metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

dataset_name = metadata["dataset_name"]
target_column = metadata.get("target_column", "")

#S5 dataset
rows = []

with open(DATASET_PATH, "r") as f:
    reader = csv.DictReader(f)

    dataset_columns = reader.fieldnames

    for row in reader:
        rows.append(row)
# S6 inspect
records_loaded = len(rows)
dataset_name = metadata["dataset_name"]
dataset_columns = reader.fieldnames
expected_records = metadata["num_records"]

print("Dataset:", dataset_name)
print("Records loaded:", records_loaded)
print("Columns (dataset):", dataset_columns)
print("Columns (metadata):", metadata["columns"])

# S7 validation
if dataset_columns == metadata["columns"]:
    column_validation = "OK"
else:
    column_validation = "MISMATCH"
print("Column validation:", column_validation)

# S8 record count validation
if records_loaded == expected_records:
    record_count_validation = "OK"
else:
    record_count_validation = "MISMATCH"
print("Record count validation:", record_count_validation)

# S9 detect invalid records
valid_records = []
invalid_records = []

for row in rows:
    temp = row.get("temperature", "").strip()
    if temp.upper() == "INVALID":
        invalid_records.append(row)
    else:
        try:
            float(temp)
            valid_records.append(row)
        except (ValueError, TypeError):
            invalid_records.append(row)

print("Valid records:", len(valid_records))
print("Invalid records:", len(invalid_records))
print("Expected invalid records (metadata):", metadata.get("invalid_records", "N/A"))

# invalid count validation
if len(invalid_records) == metadata.get("invalid_records"):
    print("Invalid record count matches metadata ")
else:
    print("Invalid record count differs from metadata")
valid_count = len(valid_records)
invalid_count = len(invalid_records)

# S10 save processed outputs
with open(VALID_OUTPUT_PATH, "w", newline="") as f_valid:
    writer = csv.DictWriter(f_valid, fieldnames=dataset_columns)
    writer.writeheader()
    writer.writerows(valid_records)

with open(INVALID_OUTPUT_PATH, "w", newline="") as f_invalid:
    writer = csv.DictWriter(f_invalid, fieldnames=dataset_columns)
    writer.writeheader()
    writer.writerows(invalid_records)

print(f"Valid records saved to: {VALID_OUTPUT_PATH}")
print(f"Invalid records saved to: {INVALID_OUTPUT_PATH}")

# S11 prepare model input
feature_columns = metadata["feature_columns"]

model_input_records = [
    {col: row[col] for col in feature_columns} for row in valid_records
]

with open(MODEL_INPUT_PATH, "w", newline="") as f_model:
    writer = csv.DictWriter(f_model, fieldnames=feature_columns)
    writer.writeheader()
    writer.writerows(model_input_records)

print(f"Model input saved to: {MODEL_INPUT_PATH}")
print(f"Records in model input: {len(model_input_records)}")
print(f"Columns in model input: {feature_columns}")


# P3: Metadata consistency check
missing_features = [col for col in feature_columns if col not in dataset_columns]
if not missing_features:
    feature_validation = "OK"
else:
    feature_validation = f"Missing feature columns: {missing_features}"

if target_column in dataset_columns:
    target_validation = "OK"
else:
    target_validation = f"Target column missing: {target_column}"

print("Feature validation:", feature_validation)
print("Target validation:", target_validation)

# S12 create ingestion summary
summary_lines = [
    f"Dataset: {dataset_name}",
    f"Records loaded: {records_loaded}",
    f"Expected records: {expected_records}",
    f"Column validation: {column_validation}",
    f"Record count validation: {record_count_validation}",
    f"Valid records: {valid_count}",
    f"Invalid records: {invalid_count}",
    "Generated files:",
    f"- {VALID_OUTPUT_PATH}",
    f"- {INVALID_OUTPUT_PATH}",
    f"- {MODEL_INPUT_PATH}"
]

with open(SUMMARY_PATH, "w") as f_summary:
    f_summary.write("\n".join(summary_lines))

print(f"Ingestion summary saved to: {SUMMARY_PATH}")
