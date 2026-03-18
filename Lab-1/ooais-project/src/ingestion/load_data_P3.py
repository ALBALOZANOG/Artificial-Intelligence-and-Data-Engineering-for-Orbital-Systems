with open("data/raw/observations.csv") as f:
    lines = f.readlines()

data = lines[1:]

temps = []
object_counts = {}

for line in data:
    parts = line.strip().split(",")
    obj = parts[1]
    temp = parts[2]

    if temp == "INVALID":
        continue

    temp = float(temp)
    temps.append(temp)

    object_counts[obj] = object_counts.get(obj, 0) + 1

avg_temp = sum(temps) / len(temps) if temps else 0

print("=== DATA SUMMARY ===")
print("Total valid records:", len(temps))
print("Average temperature:", round(avg_temp, 2))

print("\nObject occurrences:")
for obj, count in object_counts.items():
    print(f"{obj}: {count}")
