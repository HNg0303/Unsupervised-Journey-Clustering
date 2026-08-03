import json
import csv



def json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Flatten the JSON data
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
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in flattened_data:
            writer.writerow(row)

def json_to_csv_folder(folder_path, csv_file_path):
    """
        Folder of json files -> single csv file
    """
    import os

    flattened_data = []
    headers = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            json_file_path = os.path.join(folder_path, filename)
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            for item in data:
                flattened_row = flatten_json(item)
                flattened_data.append(flattened_row)

                # Update headers
                for key in flattened_row.keys():
                    if key not in headers:
                        headers.append(key)

    # Write to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in flattened_data:
            writer.writerow(row)


def flatten_json(obj, parent_key="", result=None):
    if result is None:
        result = {}

    for key, value in obj.items():
        new_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict): # Check if there is nested dict in value then key =  parent_key + key.
            flatten_json(value, new_key, result)
        else:
            result[new_key] = value

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON to CSV.")
    parser.add_argument("--json", help="Path to the input JSON file.")
    parser.add_argument("--csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    json_to_csv(args.json, args.csv)