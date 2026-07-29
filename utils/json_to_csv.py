import json
import csv

with open("behaviour_log.json", "r", encoding = "utf-8") as f:
    data = json.load(f)


def flatten_json(obj, parent_key="", result=None):
    if result is None:
        result = {}

    for key, value in obj.items():
        new_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict):
            flatten_json(value, new_key, result)
        else:
            result[new_key] = value

    return result


# Flatten all records
flattened_data = []

for item in data:
    flattened_data.append(flatten_json(item))


# Get all unique headers because different rows may have different keys
headers = []

for row in flattened_data:
    for key in row.keys():
        if key not in headers:
            headers.append(key)


# Write to CSV
with open("behaviour_log.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()

    for row in flattened_data:
        writer.writerow(row)

print("CSV file created: output.csv")